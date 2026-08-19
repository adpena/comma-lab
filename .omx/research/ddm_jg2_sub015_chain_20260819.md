# ddm_jg2 — the sub-0.15 chain: replace the modelled rate leg with a measurement

- **arm** `ddm_jg2` (task #1139 — the successor to `ddm_jg1`'s joint solve)
- **date** 2026-08-19
- **axis** every number is `[macOS-CPU advisory]` unless it carries an explicit DALI-lineage
  tag. `score_claim=false` · `promotable=false`. This arm fires **no Modal job**; MAIN owns
  the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg2/`
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.
  **Pointer UNMOVED** at contest-CUDA `0.15652626435208142` until a T4 row says otherwise.

## ANSWER FIRST

1. **The one modelled leg is now MEASURED, and the projection survives.** A re-encoder that
   reproduces the shipped token stream **byte-identically** (109,696 B, sha `15054e5d…`)
   prices jg1's 3-pair edit set at **+30 archive bytes = 4.1379 bits per changed token**.
   That is **0.877x** jg1's modelled 4.718 — the real price is **12.3% CHEAPER**.
2. **My pre-registered prediction was WRONG, and the error is instructive.** I predicted the
   real price would be HIGHER (2-5x unsurprising) and recorded that before the run. **I
   inferred MAGNITUDE from REACH.** The cascade is genuinely global — **317 of 600 frames
   carry a nonzero bit delta** — but it contributes **-4.4 bits, -1.9%: a small CREDIT.**
   101.9% of the cost is paid at the three edited frames.
3. **Edit costs SUPERPOSE.** Two extra single-pair encodes give union/sum = **1.0258**, with
   per-site interactions under 3% that change sign, and **exact** additivity at the archive
   layer (10 + 6 + 14 = 30). Causality control is exact to 0.000000 bits. So rate may be
   priced per chunk and summed — unlike seg gains and unlike compensation legs.
4. **Re-priced at the measured rate, and charging the pose leg the headline omitted:**
   seg -0.015259 · rate +0.003995 · pose +0.000314 = **net -0.010950 -> S ≈ 0.145576**,
   clearing sub-0.15 by 0.004424. **This is still a PROJECTION** — the rate leg is measured,
   the seg leg is 90 cells extrapolated 200x, the pose leg is n=3.
5. **The whole goal now rests on ONE number: the n600 realized seg yield.** Sub-0.15 needs
   it above **~1.06 cells per changed token**. jg1 measured 1.46-1.50 first-pass and
   **0.390** iterated. The stopping rule is not a refinement; it is the result.
6. **Pointer UNMOVED at 0.15652626435208142.** This arm produced an instrument, four
   measured laws, one falsified prediction of my own, and a handoff spec — **not a row.**

## THE BASE (re-read from `.omx/state/canonical_frontier_pointer.json` at arm start)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | 0.030309 |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
**Gap to sub-0.15 = 0.006526.**

## THE INHERITED PROJECTION, AND THE ONE LEG THAT IS NOT MEASURED

`ddm_jg1` (memo `.omx/research/ddm_jg1_joint_solve_20260819.md`) established, at $0:

1. a **validated** local contest-axis seg instrument (`0.99995x` of the T4 seg leg);
2. the **move-class law** — single-cell token coordinate moves repair ~1.5 argmax cells per
   changed token and compose within a sparse pass; block/dilation moves realize worse;
3. the **hard negative and its reversal** — token seg edits destroy pose (`x387`), but
   re-running the carrier's own coordinate descent against the edited frame recovers
   `d_pose` to `1.073x` of original at ~0 bytes. **The actuators compose.**

Its rate leg is **modelled, not measured**: `+4.718 bits` per changed token, computed from
the **hm1/182,759 B body's** probability model, then transferred to the to1/up3 body we
actually ship. jg1 names three reasons that constant is suspect, and **all three point the
same way — the real price is likely HIGHER**:

| # | risk (jg1 S1d caveats 3-5) | direction |
|---|---|---|
| 3 | cross-body transfer: to1's model is **sharper** (0.007446 vs 0.007603 bits/token) | costs MORE |
| 4 | context coupling: the HPAC model decodes in 190 groups, feeding decoded tokens forward | costs MORE |
| 5 | the table correction is omitted from the marginal number | unknown sign |

Two extrapolations exist and they disagree, which is itself information:

| source | repaired cells | changed tokens | net S |
|---|---:|---:|---:|
| jg1 §S1e "honest extrapolation" | ~11,400 | ~7,800 | **-0.0066** |
| jg1 §S2 first-pass scale-up (the charter's headline) | ~18,000 | ~11,600 | **-0.0104** |

The gap is 0.006526. **The honest one barely clears it; the headline clears it with room.**
Both rest on the same modelled constant. That is why S1 runs before anything else.

## STAGE LEDGER

| stage | what it settles | status |
|---|---|---|
| S1 | REAL `ΔB` for jg1's retained 3-pair edit set, through a real encoder on the pointer body | **DONE — control byte-identical; +30 B = 4.1379 bits/token = 0.877x modelled** |
| S1h | do edit costs superpose? (the density-transfer falsifier) | **DONE — union/sum 1.0258; exact at the archive layer** |
| S2 | n600 joint solve, seeded-random pair order, rate-aware acceptance | **NOT REACHED** — handed off with a binding spec below (carrier re-solve alone is 6.5-10.7 h) |
| S3 | byte-close + identity control + determinism + seal | **NOT REACHED** — gated on S2; mechanics specified below |

**HONESTY RAIL (charter, binding).** `-0.0104 S` is a 3-pair extrapolation. Realized-vs-
projected is printed at every scale rung. A smaller honest win still seals and fires; an
honest refusal with the measured curve is a first-class landing.

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at arm start) ·
`.omx/research/ddm_jg1_joint_solve_20260819.md` (full) ·
`/Volumes/APDataStore/pact/ddm_jg1/JG1_RETENTION_MANIFEST.json` + all 12 retained files ·
memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` ·
memory `the_denominator_and_the_falsifier_can_both_be_vacuous_20260816` ·
memory `concavity_helps_when_you_pay_the_axis_upward_20260818`.

---

## S1 — THE REAL RATE

### S1a — the coordinate correction, measured before anything was encoded

Two things jg1 recorded needed fixing, and both are structural rather than cosmetic:

1. **The tail is not all coder output.** `read_residual_archive` (`runtime/residual_archive.py:478-494`)
   splits it: **109,792 B tail = 96 B compact fixed residual table + 109,696 B RC64
   stream.** A re-encoder that compares its output against the whole tail is 96 B off by
   construction. This module carries the 96 B prefix through untouched and byte-checks
   only the stream.
2. **The pointer body is `ddm_up3/candidate_runtime/`, not the to1 tree.** The to1 tree's
   `inflate.py:19` pins `ARCHIVE_SHA256 = "50e56145..."` and would refuse the pointer
   archive. MEASURED here: the two archives' `runtime/` and `cpr1/` trees are identical,
   and section-wise `hpac`, `semantic` and `tail` are **byte-identical** — **only the
   carrier differs** (`f59210d7...` vs `4ef50093...`). So jg1's tail work transfers, but
   the splice target is the pointer.

### S1b — THE FINDING THAT DECIDES THE METHOD: there is no per-token price

jg1's `+4.718 bits/token` is a **per-symbol marginal**: the cost of flipping one token
holding every other probability fixed. Reading the shipped decoder
(`decode_production_tokens`, `:600-649`) shows that quantity does not exist for this
coder. **Four independent feedback paths make a token's price depend on other tokens'
VALUES:**

| # | path | file:line | reach |
|---|---|---|---|
| 1 | `sparse.selected_logits(current, context, group)` — `current` is the partially-decoded frame, one-hot'd into the conv | `residual_archive.py:617`, `cpr1/hpac_integer_sparse.py:161-170` | the 189 later groups of the SAME frame |
| 2 | `context = model.prepare_frame_context(index, previous)` — `conv_past` + SPM on the previous decoded frame | `:603`, `cpr1/hpac_integer.py:330-362` | all of frame n+1 |
| 3 | `boundary = _boundary_buckets(previous_cpu)` — the fixed-table row index is a distance-to-class-edge map of the previous frame | `:605-608`, `:515-534`, `:621` | all of frame n+1 |
| 4 | `FreeCorrector` Krichevsky-Trofimov counters, updated only from decoded symbols and **never reset per frame** | `free_corrector.py:290-308`, `173-175` | **the entire remaining stream** |

Path 4 alone makes the blast radius of one changed token **global and unbounded** — one
edit in pair 283 perturbs the probability of every symbol through pair 599. There is no
local window to price, and `ddm_hm1`'s retained `base_logits_int16_n600.i16` — which
`ddm_rr2`'s encoder memmaps — becomes **wrong** the moment a token changes. `ddm_rr2`
states its own precondition at `:40-43`: its logits are valid *"because the decoded field
is unchanged by construction."* A jg2 token edit is exactly what breaks that.

**So the modelled 4.718 was never going to be checkable by a better model. It is
replaceable only by encoding the whole 600-frame stream along the new trajectory and
stat'ing `archive.zip`.** That is what `experiments/ddm_jg2_tail_reencode.py` does: it is
`decode_production_tokens` line for line with the decode call replaced by an encode of the
known symbol, importing the model, group plan, boundary map, fixed table, corrector and
probability quantization from the shipped runtime rather than reimplementing any of them.

### S1c — the encoder is the exact inverse (smoke, n=2 frames)

The RC64 encoder half is not in the shipped tree (`runtime/entropy/rc64_backend.c`, sha
`05839d14...`, is **decoder-only**). It comes from the `ddm_rr2` lineage source pinned at
sha `5c75e2c7...` plus `route_b_rc64.RC64_CHECKPOINT_EXTENSION`. That pairing is not
assumed to invert this decoder — it is **tested**:

| quantity | value |
|---|---:|
| frames encoded | 2 |
| emitted bytes | 555 |
| **prefix bytes agreeing with the shipped stream** | **554** |
| ideal code length from the probability rows | 554.78 B |
| wall clock | 2.88 s (1.44 s/frame -> ~14.4 min for n600) |

The single trailing byte is the coder's end-of-stream flush at frame 2, which the shipped
stream does not have there. **554 of 555 bytes agree**, and the emitted length sits 0.22 B
above the ideal code length — the coder tax, measured rather than assumed. The full
600-frame control (which must be byte-identical, not prefix-identical) is the binding
proof and is running.

### S1d — PRE-REGISTERED PREDICTION, written before the n600 encode returned

Pre-registering because the honest reading of the mechanism says the modelled number
should be **too low**, and I would rather be on record than fit a story afterwards.

**The edit set.** jg1's retained 3-pair payload changes **58 tokens** (pair 283: 20,
pair 468: 19, pair 513: 19). At jg1's modelled `+4.718 bits/token` that is
**+273.6 bits = +34.2 B -> +0.0000228 S**.

**Why I expect the real price to be higher, and it is not only the three caveats jg1
listed.** jg1's own S0 measured that the stored tokens are **99.9985% identical to the
DALI GT argmax** — the shipped label field is essentially the true segmentation. The seg
actuator is therefore **PRE-DISTORTION**: it deliberately moves tokens *away* from the
natural field so that after `render -> re-segment` the argmax lands closer to GT. But the
IHS1 model is a prior fitted to the natural field. **So every edit moves a token toward
what the model finds less likely, and the four causal paths propagate that surprise
forward.** The cascade is not sign-symmetric; it should cost.

| | prediction |
|---|---|
| sign of `archive_delta_bytes` | **positive** (costs bytes) |
| realized / modelled ratio | **> 1**, and 2-5x would not surprise me |
| what falsifies "the axis survives" | realized cost above the exchange rate at jg1's 1.55 cells/token, i.e. **> ~15.8 bits per changed token** |

The measurement decides it either way, and the curve gets reported whichever way it falls.

### S1e — THE CONTROL PASSED, BYTE-IDENTICAL

| quantity | value |
|---|---|
| frames encoded | 600 |
| emitted stream | **109,696 B** |
| emitted sha256 | `15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2` |
| shipped stream sha256 | `15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2` |
| `byte_identical` | **true** (all 109,696 B) |
| wall clock | 969.8 s |

**This encoder is the exact inverse of the shipping decoder on the pointer body.** Same
model, same 190-group wavefront, same boundary map, same fixed table, same corrector, same
probability quantization, same coder, same flush. Every byte delta below is therefore a
MEASUREMENT of `archive.zip`, not a model of it.

### S1f — THE REAL RATE. MY PRE-REGISTERED PREDICTION WAS WRONG.

| quantity | value |
|---|---:|
| tokens changed | 58 (pairs 283/468/513) |
| token stream | 109,696 -> **109,726 B** |
| **archive.zip** | 176,420 -> **176,450 B** |
| **archive delta** | **+30 B** |
| **measured bits per changed token** | **4.1379** |
| jg1 modelled | 4.718 |
| **realized / modelled** | **0.877** |
| `ΔS_rate` | **+0.0000200** |

**I predicted the real price would be HIGHER than modelled, and named 2-5x as unsurprising.
It is 0.877x — 12.3% CHEAPER.** The prediction is recorded above and is not being edited.

**Where my reasoning failed, precisely: I inferred MAGNITUDE from REACH.** The four causal
paths are real and the cascade is genuinely global — the per-frame ledger shows **317 of
600 frames carry a nonzero bit delta, the first at 283 and the last at 599**, exactly as
the mechanism says. But reach is not cost:

| where the bits are paid | bits | share |
|---|---:|---:|
| at the three EDITED frames | +236.577 | **101.9%** |
| **cascade (the other 314 perturbed frames)** | **-4.409** | **-1.9%** |
| total (ideal code length) | +232.168 | 100% |

**The cascade is structurally unbounded and numerically negligible — here a small CREDIT,
not a cost.** Per-edit: pair 283 +94.2 bits, pair 468 +77.6, pair 513 +64.7; the 30 frames
after pair 513 come back **-15.4 bits**. Re-labelling a boundary cell toward the true class
makes the following frames slightly MORE predictable, and that nearly cancels the local
surprise the later edits create.

The local price also came in under the model, for the reason jg1 itself flagged: **4.718
was the mean over all four neighbour candidates, and a solver pays the accepted one.**
Measured at the sites: **4.079 bits/token**. So jg1's constant was directionally right and
mildly conservative, and its three named caveats (cross-body, context coupling, omitted
table) net out to **-12.3%**, not the multiple I expected.

| bits/token, three ways | value |
|---|---:|
| at the edit sites only | 4.0789 |
| including the whole cascade (ideal) | 4.0029 |
| **realized on `archive.zip`** | **4.1379** |

The 0.135 bits/token between ideal and realized is the coder tax — measured, not assumed.

### S1g — THE RE-PRICED PROJECTION, at the measured rate

Same first-pass rates jg1 measured (58 tokens / 90 repaired cells over 3 pairs), scaled to
600 pairs, with the rate leg now MEASURED and **with a pose charge the -0.0104 headline
omitted** (the carrier re-solve recovers `d_pose` to 1.073x, which is not free through the
sqrt):

| leg | value | S |
|---|---|---:|
| seg | 18,000 repaired cells | **-0.015259** |
| rate | 11,600 tokens @ 4.1379 bits = 6,000 B | **+0.003995** |
| pose | carrier re-solve 1.073x on `d_pose` | **+0.000314** |
| **net** | | **-0.010950** |
| **projected S** | | **0.145576** |

**That clears sub-0.15 with 0.004424 of margin — and it is still a PROJECTION.** The rate
leg is now measured; the seg leg is 90 cells extrapolated 200x, and the pose leg is n=3.

**What would have to be true to miss 0.15** (rate and pose held at their measured values,
seg yield swept):

| realized yield (cells/changed token) | repaired | net S | S |
|---|---:|---:|---:|
| **1.5517 (jg1 measured, first pass)** | 18,000 | -0.010950 | **0.145576** |
| 1.2000 | 13,920 | -0.007491 | 0.149035 |
| **1.0000 (break-even for the goal is near here)** | 11,600 | -0.005525 | **0.151002** |
| 0.8000 | 9,280 | -0.003558 | 0.152968 |
| 0.3900 (jg1's 8-pass iterated yield) | 4,524 | +0.001033 | 0.157559 |

**The goal survives if and only if the first-pass yield holds above ~1.06 cells/changed
token at n600.** jg1 measured 1.462 and 1.500 on single passes and 0.390 when it iterated
one pair to exhaustion — so the stopping rule is not a refinement, it is the whole game.
Anything that pushes past the first pass walks the score back up.

### S1h — DO EDIT COSTS SUPERPOSE? MEASURED: YES, to 2.6%.

The single biggest scale-up risk in S1f is the cross-regime one — 4.1379 bits/token was
measured at ONE edit density, and n600 is ~200x denser. The direct falsifier is whether
edit costs ADD. Two more full 600-frame encodes, each editing ONE pair:

| run | tokens | ideal bits | **archive ΔB** | bits/token |
|---|---:|---:|---:|---:|
| pair 468 alone | 19 | +76.132 | **+10** | 4.211 |
| pair 513 alone | 19 | +42.940 | **+6** | 2.526 |
| pair 283 alone (by causality, from the joint ledger) | 20 | +107.252 | +14 (implied) | — |
| **joint, all three** | **58** | **+232.168** | **+30** | **4.138** |

**Causality control, and it is exact:** the pair-468-only run's bit delta over frames 0-467
is **0.000000**, and the pair-513-only run's over frames 0-512 is **0.000000**. The model
never looks forward. That also re-proves the instrument end to end.

| | value |
|---|---:|
| naive sum of the three legs | 226.324 bits |
| **measured joint** | **232.168 bits** |
| **union / sum** | **1.0258** |
| interaction at site 468 (283 edited first) | **-1.210 bits (-1.53%)** |
| interaction at site 513 (283+468 first) | **+1.588 bits (+2.51%)** |
| archive bytes | 10 + 6 + 14 = 30 = **measured 30, exactly** |

**THE LAW: token-edit RATE costs superpose to within 2.6%, and exactly at the archive
layer.** Interactions are individually under 3% and change sign — an earlier edit can make
a later one slightly cheaper or slightly dearer, with no systematic drift.

**This is the opposite regime from `ddm_bu1`, and the distinction is load-bearing.** bu1
measured joint compensation beating the naive union by **3.705x** — that is the
COMPENSATION axis, where legs interact strongly. The RATE axis of token edits does not.
So a successor may safely price rate per chunk and sum, while it may **not** sum seg gains
(jg1 measured those decaying hard, 1.50 -> 0.390 cells/token under iteration) and may
**not** sum compensation legs. Three axes, three different additivity laws, measured.

---

## S2 / S3 — NOT REACHED, AND WHY, WITH THE WORK ROUTED

**Honest state: this arm did not produce a row. The pointer is UNMOVED at 0.15652626435208142.**
S1 was the charter's gate and it passed; S2 and S3 did not fit the unit, and saying that
plainly is the landing.

**What S2 actually costs, now that it is costed rather than assumed.** jg1 committed only
`validate` to `experiments/ddm_jg1_seg_solve.py` — its greedy composition and joint-coupling
work were ad-hoc and are not in the CLI. So S2 needs (a) a rate-aware greedy solver built on
jg1's `propose_predistortion` / `evaluate_proposal`, (b) a render + re-segment per candidate
site across 600 pairs, and (c) the carrier re-solve per edited pair, which MAIN's `ddm_na10`
budgets at **6.5-10.7 h** for n600 alone. That is more than one unit.

**Three things S2's successor inherits from this arm and must not re-derive:**

1. **The rate leg is SOLVED as an instrument, not just as a number.**
   `experiments/ddm_jg2_tail_reencode.py --stage encode` returns the exact `archive.zip`
   delta for any edited token field on the pointer body, in ~16 min, at $0, resumable, with
   a control that proves byte-identity. S2 should call it at chunk boundaries rather than
   carry a bits-per-token constant at all.
2. **4.1379 bits/token is a MEASURED CONSTANT AT ONE EDIT DENSITY** — 58 tokens sparse over
   3 pairs. n600 is ~11,600 tokens over 600 pairs, a 200x denser regime. This is the
   cross-regime-constant-transfer genus, so the constant is a PRIOR for planning and the
   re-encoder is the authority for any accept decision. The superposition probe below is the
   first evidence on that transfer.
3. **The pose leg carries a charge the headline omitted:** +0.000314 S from the 1.073x
   recovery. Per `ddm_na10` item 5 the 1.073x is n=3 and one of those pairs missed the
   relevant bar, so S2 must print the recovery DISTRIBUTION with its band on seeded-random
   pairs, never a mean. Per `ddm_na10` item 1 the PyAV-vs-DALI pose gap is **additive**
   (C = 1.4061e-04), which is why the per-pair ratio spans 0.887-1,627 and why every pose
   accept must score directly on DALI. jg1's actuator is verified DALI-clean
   (`ddm_jg1_seg_solve.py:86`).

### The rc4 rung-4 composition (MAIN's chain-extension candidate): assessed, NOT run, and re-priced downward

MAIN routed rc4's reopened token drop as a candidate stage. I did not run it, and the reason
is structural rather than scheduling:

1. **rc4 is on the hv1 body** (`S = 0.15959729…` @ **182,759 B**, archive `80d9c8c6…`), not
   our pointer (176,420 B, `7ce46fd7…`). Its `-3.243e-3 S` is exact **on that body**.
2. **rc4's mechanism is not a token-field edit — it is a RECEIVER CHANGE.** It DROPS
   positions whose model confidence exceeds a threshold and has the decoder substitute the
   prediction. My re-encoder codes a full field; it cannot skip positions without a matching
   receiver that knows which were dropped. So it is not a drop-in for this instrument.
3. **And its rate gain should be SMALLER on our body — a falsifiable prediction.** MEASURED
   here: hv1 codes the SAME token field (`9ba2e52b…`) with the SAME IHS1 blob
   (`602115b3…`) in **112,110 B**; our body codes it in **109,696 B**. Our shipped
   `FreeCorrector` already harvests **2,414 B** of exactly the redundancy rc4's
   high-confidence drop targets. **The two mechanisms compete for the same bits**, so
   rc4's 11,901 B saving cannot transfer intact, and its net `-3.243e-3` must be re-derived
   on our body before it is quoted as half the gap.

**SUPERSEDED IN PART, SAME TURN, by MAIN's second relay (`ddm_tx1`,
`.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md`).** tx1 independently re-priced rc4 on
our body at **-0.002929 S** (44.9% of the gap, 1.77x margin on the pose break-even through
the carrier re-solve) — **a 9.7% markdown from rc4's own -3.243e-3, in the direction this
section predicted from the 2,414 B corrector overlap.** Two instruments agreeing on the sign
of a cross-body transfer is worth more than either alone; I am not claiming tx1's markdown
has my mechanism as its cause, only that the direction agrees.

tx1's operative correction stands over my routing: **rc4's drops must NOT be run as a
separate composed candidate.** My single-cell EDITS and rc4's DROPS act on the same token
field in opposite directions and are **one waterfill**, sub-optimal if solved apart. Per
`ddm_bu1`'s measured law — joint compensation beat the naive union by **3.705x** — the drop
direction belongs INSIDE the S2 joint solve as a second proposal class, compensated jointly.

---

## THE S2 HANDOFF SPEC (binding on the successor)

**Objective.** A rate-aware joint descent on the pointer body that lands a byte-closed
candidate below 0.15 and hands MAIN a seal for the T4 fire.

**1. The proposal class is THREE-WAY, not two.** Per cell: `edit` (single-cell coordinate
move, jg1's measured winner) · `drop` (rc4's high-confidence prediction substitution) ·
`keep`. Solving edits alone or drops alone is the sub-optimal half of one waterfill.
Block/dilation moves are MEASURED WORSE at every radius (jg1 S1b: -55% at r=1, -351% at
r=2) and are not a proposal class.

**2. Acceptance is REALIZED and JOINT, never predicted.** Per pair: propose -> apply ->
render through the receiver's own forward model -> re-segment on the frozen CPU SegNet ->
carrier re-solve (up2's coordinate descent) against the EDITED frame -> score `d_pose`
DIRECTLY on DALI GT. Accept only on realized joint improvement. **Never** multiply an
advisory pose number by a lineage factor: the PyAV-vs-DALI gap is **additive**
(C = 1.4061e-04, `ddm_na10` item 1), which is why the per-pair ratio spans 0.887-1,627.
**Do not borrow qs5/qs1 machinery** — `qs1.GT_POSE` is still the PyAV table and optimizes
the wrong objective (`ddm_na10` item 3).

**3. The rate leg is measured, not modelled.** Call
`experiments/ddm_jg2_tail_reencode.py --stage encode` at chunk boundaries. 4.1379
bits/token is a PRIOR for planning only — it is one edit density (58 sparse tokens) and
n600 is ~200x denser. The re-encoder is the authority for any accept that turns on rate.

**4. The stopping rule IS the result.** Accept while `cells_repaired x 10.185 bits >
cost_bits` and stop at the margin. jg1 measured first-pass yield 1.46-1.50 cells/token and
**0.390** when one pair was iterated to exhaustion. Sub-0.15 needs the n600 realized yield
to hold above **~1.06**; past the first pass the rate term overtakes the seg term and the
score walks back up.

**5. Print realized-vs-projected at every rung** (n = 3 / 12 / 48 / 150 / 600) and the pose
recovery **distribution with its band**, never the 1.073x mean — it is n=3 and one of those
pairs missed the relevant bar (`ddm_na10` item 5). Budget 6.5-10.7 h for the n600 re-solve.

**6. Free lossless rider at byte-close.** The ra2+ra1 CPR1 inner coder is **lossless
-1.85e-4 S** (53x the admit bar, no scorer row, no pose budget). Its self-defeating gate
"fire only when >= 2 KB is in flight" is satisfied by this chain. Fold it into the
byte-close stage.

**7. The seal must record T4 inflate timing.** No T4 inflate-seconds figure exists for any
recent body — the 954.5 s number is **arm64 advisory and withdrawn** (`ddm_tx1` item 4).
Put "T4 inflate wall-clock READ AND RECORDED at harvest" in the receipt expectations: it is
free at harvest time and it gates the model axis both ways.

**8. Byte-close mechanics.** Splice into the POINTER member (the module verifies sha
`7ce46fd7…` before splicing and refuses otherwise). Identity control: all edits OFF must
reproduce `7ce46fd7…` byte-identically. Then double-compile determinism, container search
if brotli responds adversely (up3's 48 B lesson: archive ΔB ≠ payload ΔB — measure at the
archive layer), end-to-end `inflate` rc=0 in budget, then `candidate_seal.v1` with 8dp
falsifiers from `tac.report_8dp_bounds` (never hand-typed).
