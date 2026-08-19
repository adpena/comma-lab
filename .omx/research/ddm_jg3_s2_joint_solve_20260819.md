# ddm_jg3 — executing the S2 spec: the rate-aware joint solve

- **arm** `ddm_jg3` (task #1144 — the S2/S3 executor of `ddm_jg2`'s binding handoff)
- **date** 2026-08-19
- **axis** every number is `[macOS-CPU advisory]` unless it carries an explicit
  DALI-lineage tag. `score_claim=false` · `promotable=false`. This arm fires **no Modal
  job**; MAIN owns the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg3/`
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.
  **Pointer UNMOVED** at contest-CUDA `0.15652626435208142`.
- **code** `experiments/ddm_jg3_joint_solve.py` ·
  `src/tac/tests/test_ddm_jg3_joint_solve.py` (27 tests) ·
  `tools/ddm_jg3_rung_report.py` · commits `4ac56bc411`, `d4a0f56380`

## ANSWER FIRST

1. **No seal. The pointer is UNMOVED.** The n600 solve is launched, governed,
   resumable and checkpointed per pair, but it costs **~22 h** at full mechanism and
   did not fit this unit. Saying that plainly is the landing.
2. **Three defects in my own solver were caught by CONTROLS, not by reasoning**, and
   each one would have produced a confident wrong headline. The controls are the
   result of this arm as much as any number.
3. **The base arithmetic was wrong in the memos I inherited, and I fixed it.**
   `ddm_jg1` and `ddm_jg2` both quote `d_pose = 7.649246787e-06`. The T4 receipt
   carries **7.65e-06**, and only the receipt value reconstructs the pointer
   **bit-identically**. A test now re-derives `BASE_S` from the three components and
   refuses any drift.
4. **The hm1 logit model UNDER-PRICES a token edit by 2.2x** — 1.91 bits/token where
   `ddm_jg2` MEASURED 4.1379 on `archive.zip`. That systematically biased my
   configuration search toward too-dense edit sets. Measured, then fixed.
5. **The realized yield is ~1.0–1.3 cells/changed token, not jg1's 1.55.** That is
   above the 0.4063 accept-margin bar but **at or below the ~1.06 projection-survival
   bar**, so on current evidence sub-0.15 by token edits alone is **marginal, not
   comfortable** — which is exactly the number jg2 said the whole goal rests on.

## OPTIMAL FORM (operator binding 2026-08-19: "No naive or toy or less than recursive fractal optimal")

**Reference form** = the `ddm_jg2` S2 handoff spec, 8 clauses, in
`.omx/research/ddm_jg2_sub015_chain_20260819.md`. Every delta is declared below.

| spec clause | this arm | reduction class |
|---|---|---|
| 1. three-way proposal class `edit`/`drop`/`keep` | `edit` + `keep` implemented; **`drop` NOT implemented** | **MECHANISM — declared, with cause** |
| 1. no block/dilation moves | honored (jg1 measured them worse at every radius) | none |
| 2. acceptance realized + joint | honored, and STRENGTHENED: realized at the site level, the configuration level AND the pair level | none (superset) |
| 3. rate measured not modelled | re-encoder is the authority; the 4.1379 prior is used only to RANK configurations | none |
| 4. Lagrangian stopping rule | honored; `cells * 10.185 bits > cost_bits` | none |
| 5. realized-vs-projected at n = 3/12/48/150/600 | honored via `tools/ddm_jg3_rung_report.py` on a **shuffled** visit order | none |
| 5. pose recovery DISTRIBUTION with band | **NOT REACHED** — gated on the n600 solve | SCOPE |
| 6. ra2+ra1 CPR1 lossless rider | **NOT REACHED** — byte-close stage | SCOPE |
| 7. T4 inflate wall-clock in receipt expectations | recorded in Owed for MAIN's harvest | SCOPE |
| 8. byte-close + identity + determinism + seal | **NOT REACHED** — gated on the n600 solve | SCOPE |

**The one MECHANISM reduction, and why it is not a corner cut.** `drop` is rc4's
high-confidence prediction substitution. `ddm_jg2` measured that it is **not a
token-field edit — it is a RECEIVER CHANGE**: the decoder must know which positions
were dropped so it can substitute its own prediction, and the pointer body's
`cpr1/inflate.py` has no such path. Implementing it means shipping a new receiver,
which invalidates the byte-identity control chain the seal rests on
(`ddm_jg2` S1e/S1i: encoder reproduces the shipped stream byte-identically ->
splice reproduces `7ce46fd7…` byte-identically). Edits alone project past the gap, so
the honest ordering is to measure edits first and treat the joint edit+drop waterfill
as owed headroom. Per `ddm_bu1`'s law the joint solve is strictly better (compensation
beat the naive union by 3.705x); per the byte-close chain it costs a new receiver.
**Both are true, and this arm does not get to pretend otherwise.**

**Non-naive within-pair proposal generation.** Site order is **margin saliency**, not
`np.argwhere` row-major order. The margin field IS the Fisher surrogate (Fisher
curvature vs `(-margin)` at Pearson 0.978), and it costs nothing extra here because it
reads the same SegNet forward the argmax comes from. Candidates within a site are
ordered by the coder's own bit cost, because the objective is cells-per-bit. **A ranker
ORDERS; only realized joint ΔS accepts** — so a bad order wastes realizations but can
never put an unmeasured move into the state.

**Two bars, kept distinct** (they are different quantities and conflating them is how a
solver talks itself into a bad stop):

| bar | value | what it decides |
|---|---:|---|
| **accept-margin bar** | **0.4063** cells/token | below this, rate exactly cancels seg and NO amount of repair helps |
| **projection-survival bar** | **~1.06** cells/token | below this, the n600 projection stops clearing 0.15 |

The accept-margin bar is re-derived in code and in a test:
`RATE_PRIOR_BITS_PER_TOKEN / BITS_PER_SEG_CELL`, where
`BITS_PER_SEG_CELL = 8 * (100/117,964,800) / (25/37,545,489) = 10.1848`. MAIN
independently re-derived 10.183; the 5e-4 difference is rounding in the intermediate,
not a disagreement.

## THE BASE, RE-READ FROM THE RECEIPT (not from the charter, not from a predecessor)

| term | value | source | S contribution |
|---|---:|---|---:|
| `d_seg` | 0.00030309 | receipt `avg_segnet_dist` | 0.030309 |
| `d_pose` | **7.65e-06** | receipt `avg_posenet_dist` | 0.0087464 |
| archive | 176,420 B | receipt `archive_size_bytes` | 0.1174708 |
| **S** | **0.15652626435208142** | receipt `score` | |

Receipt: `experiments/results/modal_auth_eval_mirror/contest_auth_eval_up3_thirteenth_move_t4_r1_20260819.json`.
Archive sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
**Gap to sub-0.15 = 0.00652626.**

### CORRECTION owed to ddm_jg1 and ddm_jg2

Both predecessors quote `d_pose = 7.649246787e-06`. Substituting it, the three legs
give `0.15652583375826948` — **4.31e-07 short of the pointer**. With the receipt's
`7.65e-06` the three legs reproduce `0.15652626435208142` **exactly, to the last bit**.

The two values agree to 9.8e-05 relative, which is inside the receipt's own
3-significant-figure quote, so this is **not** a contradiction between measurements —
`7.649246787e-06` is `ddm_up2`'s higher-precision LOCAL pose instrument and is
consistent with the receipt. But **the score that ships was computed from the receipt's
number**, so pose arithmetic quoted against the pointer must use the receipt's number or
the base does not close. `test_base_S_decomposes_into_its_three_measured_legs_EXACTLY`
now asserts equality, not approximation, and
`test_base_components_come_from_the_t4_receipt_not_a_memo` re-reads the receipt.

## THE THREE DEFECTS THE CONTROLS CAUGHT

I ran my solver against `ddm_jg1`'s own pairs (283 / 468 / 513) specifically so a
number I could not argue with would tell me whether it worked. It told me three times.

### D1 — the accepted set was pooled across screening batches, and became a block move

**Symptom, measured:** pair 283 changed **66 tokens to repair 3 cells** — yield
**0.0455** — where jg1 repaired 25 from 20 (yield 1.25).

**Cause:** screening guarantees separation *within* a batch, but I pooled the winners
*across* batches, where two winners can land adjacent. A cluster of adjacent edits **is
a block move**, and `ddm_jg1` S1c measured block moves at **-55% (r=1)** and **-351%
(r=2)**. Every individual move was screened correctly and net-positive in isolation; the
joint result was still nearly worthless.

**Fix:** the accepted set carries its own separation constraint
(`select_separated`), filled greedily by net S gain. Guarded by
`test_select_separated_enforces_the_constraint_it_names`.

**Result:** pair 283 went to 9 repaired / 6 tokens = **yield 1.50** — inside jg1's
measured 1.46–1.55 band.

### D2 — a borrowed constant left half the repair on the table

Fixing D1 restored the *yield* but collapsed the *total*: 9 of pair 283's 38 flips
against jg1's 25. **Yield is not the objective; net S is, and total repair is its other
half.** jg1's ">= 64 px" additivity number came from an n=3 packing probe — a borrowed
constant, the cross-regime-constant-transfer genus.

**Fix:** the separation is **swept and measured per pair**, not borrowed. Every rung is
jointly rendered and re-segmented, and the winner is chosen on the score's own
objective. Later extended to a 2-D grid (separation × keep-fraction) because there are
two independent ways to be too greedy: spatial density (the block-move hazard) and depth
down the value ranking (the yield-decay hazard).

**Result:** pair 283 to 18 repaired / 17 tokens.

### D3 — the ranking model under-prices a token by 2.2x

With the sweep in place, the optimizer chose the **densest** rung on all three control
pairs. Back-solving its own objective from the retained sweep showed why: it was
charging **1.91 bits/token**, where `ddm_jg2` MEASURED **4.1379 bits/token** on
`archive.zip` through a byte-identical encoder.

**Cause:** I priced configurations by summing the hm1 logit costs. Those logits are the
**hm1/182,759 B generation** and our body's model is sharper — `ddm_jg1` S1d caveat 3,
named but not previously quantified. **Now it is quantified: 2.2x under-price.** An
under-priced token biases every rung comparison toward the denser configuration, because
extra tokens look nearly free.

**Fix:** configurations are priced with the constant measured on **our** body. The
logits keep their legitimate job — ranking candidates *within* a site, a relative
question on one body. Both numbers are recorded so the gap stays visible.

**This is a reusable finding, not a local bug:** any arm using the hm1 logits as an
absolute rate price on the up3 body is under-pricing by ~2.2x.

## THE MEASURED CONTROL — my solver vs jg1 on jg1's own pairs

Separation-only ladder, before the D3 pricing fix, DALI lineage, realized through the
receiver's forward model and the frozen CPU SegNet:

| pair | base flips | jg1 repaired / tokens | jg3 repaired / tokens | jg3 yield |
|---|---:|---|---|---:|
| 283 | 38 | 25 / 20 | 18 / 17 | 1.059 |
| 468 | 70 | 33 / 19 | 29 / 34 | 0.853 |
| 513 | 80 | 32 / 19 | 27 / 25 | 1.080 |
| **total** | **188** | **90 / 58 = 1.552** | **74 / 76 = 0.974** | |

**I repair 82% as many cells as jg1 and spend 31% more tokens to do it.** jg1's greedy
was never committed to its CLI, so this is not a like-for-like re-run of the same
algorithm; it is my algorithm measured against jg1's published outcome on the same
object. On this evidence **jg1's 1.552 is optimistic as a planning constant** — no arm
has reproduced it from committed code.

The retained per-pair sweep (`retained/seg_solve_ctrl3.json`) is itself the D3 evidence:
at the logit price the densest rung always won; re-scored at the measured price, pair
513 prefers separation 16 (26 repaired / 19 tokens, yield 1.368) over separation 8
(27 / 25, yield 1.080).

## THE RUNGS — realized vs projected

The n600 solve visits pairs in a **seeded permutation**, so every prefix is an unbiased
random sample and rung `n` is a genuine n-sample. This matters twice: the spec asks for
rungs, and a 22 h run can be interrupted. With the natural sorted order
(`up2.select_pairs(600)` returns `arange(600)`) every partial result would be a
**contiguous pair prefix** — the exact shape `ddm_bp2`/`ddm_na2` measured as a different
population (pose prefixes 2.54–4.21x harder, seg prefixes 0.95–0.97x easier).

Read with `tools/ddm_jg3_rung_report.py --checkpoint <jsonl>`.

**The n=3 rung, seeded-random pairs, MEASURED** (rate leg still the jg2 prior; pose at
the 1.073x recovery):

| n | flips | repaired | tokens | yield | repair% | net ΔS | projected S | clears 0.15 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **3** | 111 | **57** | **42** | **1.357** | **51.4%** | **−0.006457** | **0.150069** | **NO — by 6.9e-05** |

**That is the headline of this arm, and it is a knife edge.** Token edits alone land
**0.000069 ABOVE 0.15** — about one part in 2,200 of the score. The yield (1.357) is
comfortably above both bars; the repair fraction (51.4%) is close to what jg1's
extrapolation assumed. The move very nearly clears on its own and does not.

**Which makes the free lossless rider DECISIVE rather than optional.** The ra2+ra1
CPR1 inner coder is **MEASURED at −1.85e-4 S**, lossless, round-trip exact, $0, with no
scorer row and no pose budget consumed. Applied to this rung:

> **0.150069 − 0.000185 = 0.149884 — clears sub-0.15.**

Spec clause 6 called it a "free lossless rider at byte-close." On this evidence it is
not a rider; it is **the margin**. Any successor that treats it as optional polish will
miss the goal by less than one part in two thousand.

**Read this rung for what it is.** n=3 is three pairs. The per-pair spread is wide
(yields 1.286 / 1.357 / 1.500 across the first rungs measured), the rate leg is the
sparse-density prior, and the pose leg is jg1's n=3 mean. It is quoted because it is
measured and because it decides where the next unit points — not because three pairs
settle a 600-pair question.

## WHAT WOULD HAVE TO BE TRUE

Rate and pose held at their measured values, seg yield and repair fraction swept. The
projection is `net = -R * 8.4771e-07 + (R/y) * 3.4443e-07 + 0.0003136`, where `R` is
cells repaired at n600 and `y` is the realized yield:

| realized yield | repair needed to clear 0.15 | as a share of the 35,754 flips |
|---:|---:|---:|
| 1.552 (jg1's published first pass) | 10,930 | 30.6% |
| 1.060 (the projection-survival bar) | 13,084 | 36.6% |
| 0.974 (**this arm's measured control**) | 14,159 | 39.6% |
| 0.800 | 16,394 | 45.8% |
| 0.600 | 24,995 | 69.9% |
| 0.4063 | impossible at any repair | — |

## THE CHAIN IS PROVEN END TO END, AND THE PAYLOAD CLAIM IS CHECKABLE

Two things that de-risk MAIN's harvest, both verified rather than asserted:

1. **The per-pair checkpoint IS the payload.** `tools/ddm_jg3_edits_from_checkpoint.py`
   rebuilds the edited token planes from the JSONL's sparse `(y, x, value)` records
   against the sha-pinned base field, and **refuses if the reconstructed token count
   disagrees with the count the checkpoint recorded**. So "always keep the payload" is
   a checked property here, not a claim — the bytes are recoverable at any moment,
   including mid-run, without waiting for the npz mirror.
2. **solve -> reconstruct -> re-encode -> `archive.zip` delta runs end to end.** The
   reconstructed edit set was fed straight into `ddm_jg2_tail_reencode.py --stage
   encode` on the pointer body. That is the exact path a seal must take, exercised
   before anyone needs it under time pressure.

### And the instrument REFUSED TO CERTIFY ITS OWN NUMBER, which is the best thing it did

The encode returned a delta **and flagged it untrustworthy**:

```
{"event": "UNPROVEN", "note": "payload retained, but the 600-frame control has not
 proved this encoder byte-identical on this body; the delta is NOT trustworthy"}
{"archive_delta_bytes": 31, "delta_trustworthy": false, "control": null, ...}
```

`ddm_jg2` ran its byte-identity control **in its own store**; my store carries no
control record, so the encoder fails closed rather than letting me quote a number whose
instrument was never proven here. **That is the vacuity-equals-pass failure being
actively prevented** — a skipped control would otherwise have read as green.

The numbers, quoted as UNPROVEN and not as a result:

| quantity | value |
|---|---:|
| tokens changed (6 pairs) | 68 |
| token stream | 109,696 -> 109,727 B |
| `archive.zip` | 176,420 -> 176,451 B (**+31 B**) |
| bits per changed token | **3.6471** |
| vs jg2's MEASURED 4.1379 | **0.881x** |
| `delta_trustworthy` | **false** |

Read carefully, this is mildly *encouraging* and firmly *not evidence*: at ~17% higher
edit density than jg2's probe the implied price went **down** (3.65 vs 4.14 bits/token),
where the cross-regime worry was that it would go up. But 68 tokens over 6 pairs is
still the sparse regime, not the ~11,000-token regime an n600 solve produces, and the
instrument itself says the delta is not trustworthy on this store.

**Owed, and cheap: `ddm_jg2_tail_reencode.py --stage control` in the jg3 store — ~16 min,
$0 — flips `delta_trustworthy` to true and makes every later rate number quotable.**
It was not run first here, and it should have been; that ordering is now in Owed.

**The n600 solve is safely detached.** PID 81175 has been reparented to `launchd`
(PPID 1), so it survives this session; it checkpoints every pair and resumes with
`--resume`. Rungs are readable at any moment.

## Owed, with owners

0. **Run `ddm_jg2_tail_reencode.py --stage control --store <jg3 store>` FIRST**
   (~16 min, $0). It proves the encoder byte-identical on this body and flips
   `delta_trustworthy` to true. Every rate number this arm or a successor quotes from
   the jg3 store is UNPROVEN until it runs. I ran the encode before the control; that
   ordering was wrong and the instrument caught it.
1. **The n600 solve must finish** (~22 h from launch, resumable, per-pair checkpoints
   at `/Volumes/APDataStore/pact/ddm_jg3/checkpoints/seg_solve_n600.jsonl`). Rungs are
   readable at any time. **Owner: MAIN's harvest, or a successor arm with `--resume`.**
2. **Then the rate leg becomes MEASURED, not prior**: one
   `experiments/ddm_jg2_tail_reencode.py --stage encode` on the final edit set, ~16 min,
   $0. jg2 measured rate costs superpose (union/sum 1.0258, exact at the archive layer),
   so a chunked measurement is legitimate.
3. **Then the pose leg**: `up2.solve_pair_realized` against the EDITED frames, budget
   6.5–10.7 h, printing the recovery **distribution with its band** — never the 1.073x
   mean, which is n=3 and one of those pairs missed the relevant bar.
4. **Byte-close (spec clause 8)** with the `ddm_cw1` tie-break pinned: configs 5 and 6
   tie at 176,420 B and **only the lower index reproduces the shipped bytes**, so
   equal-byte ties must break to the lower index or the identity control flickers.
   `src/tac/win_families/` + `experiments/ddm_cw1_container_consumer_proof.py` is the
   engine that byte-reproduced `7ce46fd7…` on the real body.
5. **The `drop` proposal class** — the declared MECHANISM reduction. Needs a receiver
   that knows which positions were dropped. This is the one-waterfill headroom
   `ddm_bu1`'s 3.705x law says is being left on the table.
6. **T4 inflate wall-clock READ AND RECORDED at harvest** (spec clause 7). No T4
   inflate-seconds figure exists for any recent body; the 954.5 s number is arm64
   advisory and withdrawn.

## My own round-1 adversarial review

1. **Did I quote a prefix anywhere?** No, and I had to build the cure: the n600 visit
   order is a seeded permutation precisely because `up2.select_pairs(600)` returns
   sorted `arange(600)` and every rung would otherwise be a contiguous prefix. The
   within-pair `--max-sites` subsample is also seeded-random rather than a head slice,
   because frame rows are strongly class-skewed (Undrivable/sky rows ~9–182, MyCar rows
   ~290–379) so a head slice would sample a different edge mixture.
2. **Is my instrument circular?** The seg objective is `ddm_jg1`'s, which reproduces two
   independently published legs on two lineages (0.99995x DALI, 1.00002x PyAV) with a
   byte-exact forward model. I did not re-derive it and I did not re-tune it.
3. **Did I catch my own errors?** Three, all by control rather than reasoning (D1/D2/D3),
   plus the inherited `d_pose` base error caught by a test that demanded EXACT
   reconstruction rather than approximate. The lesson is the one the campaign already
   knows and I needed anyway: **make the control demand equality, not closeness.**
4. **Is the batched-SegNet speedup science-neutral?** Verified by identity, not argued:
   the same 6-site smoke returned repaired 3 / tokens 5 / residual 1 at both batch 1 and
   batch 8, in 32.9 s vs 18.6 s. BatchNorm is in eval mode and reads running statistics.
   The RENDER stays at batch 1 because `ddm_up2` sec.6 measured semantic batch 8 as
   byte-changing.
5. **What is the weakest thing I lean on?** The rate leg. 4.1379 bits/token is measured
   on `archive.zip`, but at ONE edit density — 58 sparse tokens over 3 pairs — and n600
   is ~200x denser. jg2's superposition law (union/sum 1.0258) is real evidence that it
   transfers, but it too was measured sparse. **Nothing here is byte-closed.**
6. **Did I over-claim the control?** The 3-pair comparison uses jg1's CHOSEN pairs
   (38/70/80 flips against a population mean of 59.6), not a random sample. It is a
   reproduction control, labelled as such, and it is not a population estimate. The
   population estimate is the shuffled n600 rung table.
7. **What did I not do.** No seal, no byte-closed archive, no re-encoder run, no pose
   re-solve, no `drop` class, no n600 completion. **The pointer is UNMOVED at
   0.15652626435208142.**

## Retained payload

`/Volumes/APDataStore/pact/ddm_jg3/` — per-pair checkpoints (JSONL, which carry the
edit payload **losslessly** in sparse `(y, x, value)` form against a sha-pinned base
token field), the edited token planes as npz, the per-pair separation sweeps, and the
run logs. Payloads are persisted every pair, never only their measured lengths.
