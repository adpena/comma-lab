# ddm_rq1 — the shipped realization is a conditional optimum of the JOINT problem: coarsening it loses 148–2,641× and refining it loses too. Stage B does not run.

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · promotion_eligible=false ·
pointer_moved=false

<!-- # FORMALIZATION_PENDING: the law this arm proposes -- that the deployed member's
REALIZATION (its depth table) is held at a conditional optimum by the token field rather than by
the weights, so a realization change loses in BOTH directions -- is n=1 on ONE object and ONE
field. It is registered as a canonical equation only when a second object reproduces it, per the
binding-numbers-expire discipline. The variant-table rows and the two corrections in §5 are
BYTE-IDENTITY and n600 MEASUREMENT receipts, not models, and bind immediately. -->

Base: move 48, S 0.13638261682704697 @ 179,111 B `[contest-CUDA T4 n600]`, archive
`d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`; components d_seg 0.00010345,
d_pose 4.59e-06; admit bar −2e-5. Field `a92e7d90…`. HEAD at arm start `14cc70a63`.

---

## ANSWER FIRST

**The pointer did not move. Stage B did not run, nothing is staged, MAIN fires nothing.**

**The control PASSED on both legs, exactly.** Re-encoding the shipped float weights at the shipped
realization reproduces the shipped semantic member **byte-for-byte** (29,862 B, sha
`786950a5b0b2e746…`), moves **0** argmax cells against `ddm_ren1`'s retained shipped plane, and
reproduces the n600 `d_seg` to **seventeen significant figures**
(`0.00010338677300347222`, identical to `ddm_ren2`'s control). The pricer is the shipped encoder.

**Stage A's verdict, and it is not close:**

> The realization axis buys rate at **116 to 2,075 argmax cells per byte**. The exchange rate at
> move 48 allows **0.7855 cells per byte**. Every coarsening rung is therefore **148× to 2,641×**
> underwater, and the cheapest rung still needs a **99.32 %** repair fraction against a measured
> best-ever single pass of **40.39 %** and a measured best-ever CUMULATIVE multipass of **45.8 %**.

**And the finding that makes it a law candidate rather than a disappointment: the loss is
two-sided.** The shipped object already realizes two tensors at THREE bits. Promoting them back to
four — spending bytes to buy accuracy — also **raises** `d_seg`:

> `blocks.0.film.weight` 3→4 bits costs **+22 B** and **+881 net wrong cells** (+7.468e-4 S).
> Both tensors 3→4 costs **+77 B** and **+1,189 net wrong cells** (+1.008e-3 S).
> `ddm_ntb2`'s named prize — "recover the +4.92e-05 S the 3-bit `blocks.0.film` choice cost" — is
> **FALSIFIED on this object**: the recovery is not +4.92e-05 S, it is **−7.468e-4 S**, 15.2× in
> the wrong direction.

The mechanism is the one `ddm_ren2` located from the other side. Moves 24–48 solved 600 token
planes against *this exact realization*, quantization error included. The 3-bit rounding on
`blocks.0.film` is not an error the field is suffering; it is part of the pre-image the field was
built against. `ddm_ren2` measured that moving the renderer's **values** loses and is no better
than a coin flip of the same size. This arm measures the complementary half: moving the renderer's
**grid** loses in **both signs**, and the per-byte loss does not improve as the rung shrinks — it
gets **worse** (the smallest rung, −16 B, is the worst in the table at 2,075 cells/B).

**Stage B is priced shut a second time, independently of any repair fraction.** Even granting a
100 % repair, the repair must be PAID FOR in token bytes. At the most favourable measured
efficiency (1.308 cells per changed token, pass 1) and the cheapest measured marginal cost
(5.2537 bits per changed token, pass 6), the cheapest rung's repair bill is **111,457 B** of token
stream to buy back **1,914 B** — **58.2× underwater** — and the largest rung's is **524,935 B**,
which is **2.93× the entire 179,111 B archive**.

---

## 1. The instrument and the control (STOP gate: PASSED)

Producer `experiments/ddm_rq1_coarsen_and_price.py` (sha `40cf4ddbe56fd0c7…`), which REUSES
`ddm_ren2`'s landed producers rather than re-implementing them: `ddm_ren2_restore_init.encode_member`
(the pack → SM1S → CK2 → Brotli chain), `ddm_ren2_price_checkpoint.load_shipped` /
`renderer_with_state`, `ddm_ren1`'s retained shipped argmax plane, `ddm_jg1`'s n600 seg instrument,
and `ddm_up2.LINEAGE_DALI` for the GT table. The float weights come from `ddm_ren2`'s restored init
(`init_restored.pt`, sha `ba51f59f…`, READ ONLY) and **never change**: this arm trains nothing.

| control leg | measured | reference | agreement |
|---|---|---|---|
| SM3R body re-encode | 36,130 B `17e0fd0b…` | shipped | **byte-identical** |
| SM1S rider re-encode | 31,451 B | shipped | **byte-identical** |
| RX1M member re-encode | **29,862 B `786950a5b0b2e746…`** | shipped | **byte-identical** |
| n600 `d_seg` through R | **0.00010338677300347222** | `ddm_ren2` `0.00010338677300347222` | **17 figures** |
| cells moved vs shipped plane | **0** | — | exact |

The byte-identity gate runs on **every** invocation, not only on the control label, and is a
refusal: a variant's member delta is a price only if the same encoder reproduces the shipped member
exactly on the shipped realization in the same process.

**The instrument's reproduction gap, in this arm's own units.** The pointer's T4 row is
`d_seg 0.00010345`; this instrument's control is `0.00010338677300347222`. The gap is
**7.46 cells = 6.323e-06 S = 0.316 admit bars**. The smallest effect measured below (881 net cells)
is **118×** that gap, so every row in the table resolves by a wide margin.

---

## 2. The variant table (n600, MEASURED, at the FIXED field `a92e7d90…`)

All eight priced realizations come from the **same float weights** with only the SM3R header's
depth table or `keep_percent` changed — both fields the receiver reads out of the bytes it is
handed. **No receiver code is touched; the member's VALUES are the whole change.**

| variant | what moved | member B | Δ B | rate bought (S) | n600 `d_seg` | cells moved | damaged | repaired | **net cells** | debt (S) | net ΔS @ 0 repair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **control / (e)** | nothing | **29,862** | **0** | 0 | **0.00010338677300** | **0** | 0 | 0 | **0** | 0 | 0 |
| **(a)** all fourteen 4-bit → q3 | 14 tensors | 21,863 | **−7,999** | −5.3262e-3 | 0.00896648 | 1,054,511 | 1,049,741 | 4,208 | **+1,045,533** | +0.886309 | **+0.880983** |
| **(b)** largest eight → q3 | 8 tensors | 22,321 | −7,541 | −5.0212e-3 | 0.01131616 | 1,332,313 | 1,327,100 | 4,387 | **+1,322,713** | +1.121278 | **+1.116256** |
| **(c)** head + blocks.3 → q3 | 3 tensors | 27,948 | −1,914 | −1.2745e-3 | 0.00198525 | 230,079 | 225,886 | 3,892 | **+221,994** | +0.188187 | **+0.186912** |
| **(f)** largest single → q3 | `coord_mix` | 28,453 | −1,409 | −9.3820e-4 | 0.00838760 | 986,074 | 981,184 | 3,938 | **+977,246** | +0.828422 | **+0.827484** |
| **(g)** smallest single → q3 | `token_embed` | 29,846 | **−16** | −1.0654e-5 | 0.00038476 | 37,312 | 35,210 | 2,018 | **+33,192** | +0.028137 | **+0.028127** |
| **(h)** both 3-bit → q4 | `frame_embed`+`blocks.0.film` | 29,939 | **+77** | +5.1271e-5 | 0.00011347 | 1,306 | 1,246 | 57 | **+1,189** | +0.001008 | **+0.001059** |
| **(i)** `blocks.0.film` → q4 | 1 tensor | 29,884 | **+22** | +1.4649e-5 | 0.00011086 | 927 | 902 | 21 | **+881** | +0.000747 | **+0.000755** |

(d) is **NOT EXPRESSIBLE** — §5.2. "Net cells" is against the instrument's own control, not the T4
row, so no cross-instrument subtraction is hidden. Every member byte number is a **REAL encode**
through the shipped container (Brotli q10 / lgwin16 + CK2), twinned by the control identity in the
same process. Every payload — `body.sm3r`, `rider.sm1s`, `member.rx1m` — is retained per variant.

### 2.1 The exchange, in one number per rung

`1 B = 6.658589531221714e-7 S`; `1 cell = 100/117,964,800 S = 8.4771e-7 S`; so the exchange allows
**0.7855 cells per byte**. Measured:

| variant | cells per byte moved | × over the allowance | required repair fraction to break even | lotteries (34.8 B) in the wrong direction |
|---|---:|---:|---:|---:|
| (c) | **116.0** | **148×** | **99.323 %** | 8,066 |
| (a) | 130.7 | 166× | 99.399 % | 38,020 |
| (b) | 175.4 | 223× | 99.552 % | 48,173 |
| (f) | 693.6 | 883× | 99.887 % | 35,711 |
| (g) | **2,074.5** | **2,641×** | 99.962 % | 1,214 |
| (h) | 15.4 (spent, not bought) | — | **105.09 % — impossible** | 45.7 |
| (i) | 40.0 (spent, not bought) | — | **101.96 % — impossible** | 32.9 |

The refinement rungs' break-even fractions exceed 100 % because a refinement pays bytes AND cells:
no repair fraction can make it profitable, because even a perfect repair leaves the byte cost.

**The curve does not bend toward profitability as the rung shrinks.** That was the one way this
family could have survived — a rung small enough to sit inside the budget — and it is measured
shut: the smallest rung in the table (−16 B) is the **worst** per byte by 2.6× over the best, and
the best per byte (116.0) is still 148× the allowance. The ranking is tensor-dependent, not
size-dependent, which also inverts `ddm_rw1`'s naming: its "most sensitive" set (c) is the
**least** damaging per byte here, and `coord_mix` — not in that set — is 6.0× worse.

### 2.2 The class split of the damage, and an independent reproduction of the Lane law

Damaged cells split by the GT class at the damaged site, expressed as over-representation against
each class's measured n600 area share (road 23.2 %, lane 0.59 %, undrivable 49.5 %, movable 1.24 %,
mycar 25.4 %):

| variant | road | **lane** | undrivable | movable | mycar |
|---|---:|---:|---:|---:|---:|
| (a) | 3.6× | **12.8×** | 0.1× | 2.4× | 0.0× |
| (c) | 2.7× | **34.7×** | 0.1× | 8.4× | 0.0× |
| (g) | 2.6× | **20.8×** | 0.4× | 4.2× | 0.0× |
| (h) | 1.7× | **46.0×** | 0.4× | 8.8× | 0.2× |
| (i) | 1.7× | **44.3×** | 0.4× | 9.7× | 0.1× |

**Lane is 12.8–46.0× over-represented in the damage, and the over-representation RISES as the
perturbation shrinks** — 44–46× at the two smallest rungs. That reproduces the campaign's standing
"Lane 40×" reading from a completely different actuator: every prior measurement of it came from
token moves, this one comes from changing the renderer's quantization grid. Undrivable (0.0–0.4×)
and MyCar (0.0–0.2×) are essentially untouched, which is the #139 static-core signature.

---

## 3. The repair fractions this arm cited, and their receipts

Read out of the receipts, not recalled. All four are `[contest-CUDA]`-sealed or n600-instrument
rows of the SAME actuator Stage B would have used.

| pass | receipt | flips before → after | repaired | tokens changed | cells / changed token | bits / changed token (realized) | break-even bits/token |
|---|---|---|---:|---:|---:|---:|---:|
| pass 1 (jg1/jg3) | `.omx/research/ddm_sj1_multipass_token_predistortion_20260905.md` §"forward-model control" (sha `3d848e59…`) | — | 12,003 cells | 9,179 | **1.308** | — | — |
| **pass 2a** (first full pass, fresh object) | same memo §5, field `e107b5ab…` | 23,749 → 14,156 | **9,593 = 40.39 %** | 7,804 | 1.229 | 6.2307 | 12.52 |
| pass 3 | `.omx/research/ddm_sj1_t4_token_predistortion_pass3_20260906_pointer_move_32_20260906.md` (sha `ab14594f…`) | 14,157 → 12,866 | **1,447 = 10.22 %** | 1,339 | 1.081 | 5.31 | 11.0 |
| pass 6 | `.omx/research/ddm_sj1_t4_token_predistortion_pass6_20260910_pointer_move_43_20260910.md` (sha `d57ab4d5…`) | 12,540 → 12,196 | **344 = 2.74 %** | 335 | ~1.03 | **5.2537** | 10.4585 |

**Best measured single pass: 40.39 %. Best measured CUMULATIVE (passes 2a + 3): 45.8 %**
(23,749 → 12,866). The family re-opens per object when planes are re-rendered — pass 6 measured
375 of 489 admitted positions new on the 365 re-rendered pairs and **zero** new on the 235
unchanged pairs — so a coarsened renderer would indeed re-open it fully. That is the charter's
bet, and it is not enough: the smallest demand in §2.1 is **99.32 %**, which is **2.2×** the best
cumulative fraction the actuator has ever produced, on a pool **86×** larger than any it has ever
worked on (1,045,533 cells vs the 23,749 it started from).

### 3.1 The repair-fraction-independent closure (the one that settles it)

Grant a 100 % repair fraction for free. The repair still costs token bytes. Using the most
favourable measured efficiency in the table above (1.308 cells/changed token, pass 1) and the
cheapest measured marginal cost (5.2537 bits/changed token, pass 6) — i.e. two numbers chosen to
flatter the bet, each from a different pass:

| variant | net cells to repair | tokens needed | token-stream bytes | vs the bytes the rung buys | vs the whole 179,111 B archive |
|---|---:|---:|---:|---:|---:|
| (c) | 221,994 | 169,720 | **111,457 B** | **58.2× underwater** | 0.62× |
| (a) | 1,045,533 | 799,337 | **524,935 B** | **65.6×** | **2.93×** |
| (b) | 1,322,713 | 1,011,248 | 664,100 B | 88.1× | 3.71× |
| (f) | 977,246 | 747,130 | 490,650 B | 348.2× | 2.74× |
| (g) | 33,192 | 25,376 | 16,665 B | 1,041.6× | 0.09× |

Two of the five rungs would need a token stream **larger than the entire archive** to undo their
own damage. This closure does not depend on the repair fraction at all, so it cannot be argued away
by "run more passes".

---

## 4. THE STOP DECISION

**Stage B does not run.** The charter's rule is: Stage B runs only if some variant's seg debt does
not exceed its rate gain even at a 100 % repair fraction, and otherwise pick the variant maximising
`rate_gain − debt × (1 − r_repair)`. Applied with `r_repair` from the measured pass-1/2 fractions
above (**0.4039** best single pass; **0.458** best cumulative), every variant is refused, and the
best of them by a factor of 96:

| variant | rate term ΔS | debt (S) | net ΔS `= rate + debt × (1 − 0.458)` | admit bars |
|---|---:|---:|---:|---:|
| (i) | +1.46489e-5 | +0.000747 | **+0.000419** | **+21.0** |
| (h) | +5.12711e-5 | +0.001008 | **+0.000598** | +29.9 |
| (g) | −1.06537e-5 | +0.028137 | **+0.015240** | +762.0 |
| (c) | −1.27445e-3 | +0.188187 | **+0.100723** | +5,036.1 |
| (f) | −9.38195e-4 | +0.828422 | **+0.448066** | +22,403.3 |
| (a) | −5.32621e-3 | +0.886309 | **+0.475053** | +23,752.7 |
| (b) | −5.02124e-3 | +1.121278 | **+0.602711** | +30,135.6 |

Nothing is positive-signed in our favour; the closest row is 21 admit bars the wrong way. Even
setting `r_repair = 1.0` exactly (physically impossible, and separately refused by §3.1's token
bill), the two refinement rungs still lose, because a refinement pays bytes as well as cells.

**No Stage B was launched. No archive was built, no carrier was re-solved, no seal inputs were
staged, no packet was written, no Modal call was made.** Deliverable 5 of the charter is
conditional on a surviving variant and none survived, so spending a terminal n600 carrier re-solve
on a row its own cheapest binding term has already refused by 21 to 30,000 admit bars would have
been the means-as-ends failure, not diligence.

---

## 5. CORRECTIONS OWED UPWARD

### 5.1 `ddm_ntb2`'s named prize is FALSIFIED on this object (MEASURED)

`ddm_ren2` §7.3 records it as still open: *"Recovering the `+4.92e-05 S` that the 3-bit
`blocks.0.film` choice cost is 0.48 %."* Measured here by the only action that could collect it —
realizing that tensor at four bits instead of three, same float weights, shipped packer, shipped
container, n600 through R:

| | claimed by `ddm_ntb2` (via `ddm_ren2`) | **MEASURED here** |
|---|---:|---:|
| seg effect of `blocks.0.film` 3→4 | −4.92e-05 S (a recovery) | **+7.4683e-04 S (a cost)** |
| net wrong cells | −58 (repaired) | **+881 (damaged)** |
| member byte cost | not priced | **+22 B** (real encode) |
| net ΔS | −4.92e-05 (would clear the bar) | **+7.615e-04 (32.9 lotteries the wrong way)** |

Sign inverted, magnitude 15.2× off. The prize is not merely smaller than claimed; it does not
exist on this object at this field. A successor must not spend a unit collecting it.

**Why, and this is the transferable part.** The claimed cost was derived on an earlier object where
the field had not been solved against the 3-bit choice. On the move-48 object the field HAS been:
moves 24–48 pre-distorted 600 token planes against this exact realization. The 3-bit rounding is
inside the pre-image, so removing it is a perturbation like any other. This is the
`binding-numbers-expire` genus applied to a DERIVED quantity: `ntb2`'s number was true of the
object `ntb2` measured.

### 5.2 The charter's variant (d) is not expressible in the shipped format (MEASURED)

Receipt: `/Volumes/VertigoDataTier/pact/ddm_rq1/stageA/member_d_keep_percent_deeper/FILM_PRUNE_FLOOR.json`.
`pack_prune_mixed_candidate` refuses `keep_percent` outside `[1, 99]`, the shipped value is already
**1**, and the packer's own rule `keep = max(1, round(rows × keep_percent / 100))` gives
`max(1, round(192 × 0.01)) = 2` kept rows — so **no admissible `keep_percent` realizes fewer rows
than the object already keeps**. Deepening the FiLM prune is a RECEIVER change, out of this arm's
bounds. And it would not be worth one: all three pruned FiLM tensors together hold **24 B** of
codes plus 72 B of masks, so the entire family is ~1.3 admit bars even if deleted outright.

### 5.3 The charter's variant (e) is a no-op by construction, proved by byte identity

"(e) the shipped depths with the per-axis scale re-fit (control for the packer)" re-encodes to
**29,862 B sha `786950a5b0b2e746…`, byte-identical to the control.** `standard_qn_payload`
recomputes the per-axis fp16 scales from the float tensor on **every** call, so there is no stored
scale to preserve and no re-fit to perform: "re-fit the scales" and "encode at the shipped depths"
are the same operation. The row is kept in the table as a zero-drift control on the packer, which
is what the charter wanted from it, and it reports zero.

### 5.4 One about this arm's own producer, recorded rather than hidden

The first version of `ddm_rq1_coarsen_and_price.py` persisted the member's LENGTH and sha while
holding the encoded bytes in memory — the measure-and-discard class CLAUDE.md forbids at the typing
moment. Caught in this arm's own round-2 self-review before any conclusion was drawn from it, and
fixed by retaining all three encode stages (`body.sm3r` / `rider.sm1s` / `member.rx1m`) for every
variant; every row in §2 was then re-emitted by the fixed producer, resuming the identical
`seg_rows.jsonl` (row shas unchanged), so no number changed and every payload is now on the tier.
The retained control member re-hashes to `786950a5b0b2e746…`, the shipped one.

Two smaller provenance notes: three `--legs member seg` rows (a/b/c) were first produced by an
earlier revision of the producer and re-emitted by the final one from the same resumed rows, so
each `INPUTS.json` records the producer sha that actually ran; and variant (f)'s "largest single
tensor" resolved to `coord_mix.weight` (9,600 elements) by measurement from the template, not by
hand-ranking.

---

## 6. WHAT IS CLOSED, AND AT WHAT SCOPE

**The member's REALIZATION axis is closed at FAMILY scope on this object and this field.** The
format exposes exactly two realization fields to a non-receiver change — the per-tensor depth table
and `keep_percent`. Both are measured: `keep_percent` is at its floor and not expressible deeper;
the depth table loses in both directions, at 116–2,075 cells per byte against an allowance of
0.7855, over a rung range spanning 500× in coarsened values (480 to 53,040) and both signs.

**What makes this more than "eight variants failed":** the per-byte ratio was measured across that
whole range and does **not** improve toward small rungs — it degrades. The escape route this family
had (find a gentle enough rung) is measured shut rather than argued shut, which is the same move
`ddm_ren2` made with its amplitude-matched random-direction control.

**What is NOT closed:**
* **Per-row mixed depth** — a receiver change. `ddm_ren2` named it; this arm's measurements make it
  a *worse* bet than it looked, because the loss is realization-change-shaped rather than
  precision-shaped: (i) shows that even ADDING precision to one tensor costs 881 cells.
* **The alternating joint solve** (`ddm_ren2` §7.2): refit θ, then RE-SOLVE the field against the
  new θ. This arm's result sharpens the prior on it in both directions. In its favour: the field
  really is what holds θ in place, which is exactly why re-solving it is the mechanism that could
  move θ. Against it: the debt a θ change incurs is 116–2,075 cells/B, while the field's repair
  actuator delivers ~1.2 cells per changed token at ~5.3 bits/token — so the alternation has to
  start from a θ change orders of magnitude gentler than anything this format can express, which
  §2 measures does not exist on the depth axis and `ddm_ren2` §3 measures does not exist on the
  amplitude axis either (no nonzero rung below 0.143 LSB).
* **The token/field axis itself**, which is untouched by this arm and is where moves 24–48 came
  from. Pass 6's convergence rule (< 1 % of remaining flips per pass) was NOT reached at 2.74 %.

### What the next unit should run, specified

1. **Do not price another realization rung of this format.** Eight rungs, both signs, a 500× size
   range, one n600 row each. A ninth is not new information.
2. **Do not spend a unit on `ddm_ntb2`'s prize.** §5.1 measures it in the wrong direction.
3. The live doors are the token/field axis (pass 7 on a successor field that re-renders pairs) and
   the two objects `ddm_sj1` pass 6 listed as next: the counted-rider cure re-based onto the
   shipping bytes, and the frame-0 selector splice (+0.075 bar, a build item).
4. **Reuse, do not rebuild:** `ddm_rq1_coarsen_and_price.py` prices ANY realization of this member
   end to end — real encode, parse-back, n600 `d_seg` through R, cells moved/damaged/repaired, the
   class split, the whole-score projection — in about six minutes on this box, with the byte-identity
   gate firing on every invocation. All eight realizations' packed bytes are retained.

---

## 7. BOUNDARIES AND CUSTODY

- **No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`, no paid dispatch, no contest
  evaluation, no seal call, no packet, no candidate archive, no seal inputs staged.** MAIN fires
  everything. Nothing here is a score.
- Nothing edited under `upstream/`, `submissions/semantic_joint_ctxmix/`, any sealed tree,
  `src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, or any **receiver code**. The only
  thing this arm ever changed is the semantic member's VALUES — its depth table and `keep_percent`,
  both fields the shipped receiver reads out of the bytes — and no candidate member was shipped.
- **Nothing was trained.** The float weights are `ddm_ren2`'s restored init, unchanged, bound by
  sha; only the grid they are realized on moved.
- `ddm_ren1`'s, `ddm_ren2`'s, `ddm_sj1`'s and `ddm_dpi1`'s directories were **read only**
  (`ddm_ren2`'s `init_restored.pt`; `ddm_ren1`'s retained `planes.npz` for the cells-moved
  decomposition; `ddm_sj1`'s retained token field). Nothing was written into any of them.
- **No storage reserve lowered.** Every write goes through a 40 GiB fail-closed reserve check.
- **Every payload on the SSD tier** under `/Volumes/VertigoDataTier/pact/ddm_rq1/`, sha-verified in
  `RETENTION.json`: **150 payloads, 2,596,825 B = 0.00242 GiB** against the 8 GiB cap, 52.5 GiB free
  after, reserve respected. That includes all three encode stages of all eight realizations, every
  per-pair `seg_rows.jsonl`, every `INPUTS.json` / `RESULT.json`, and the `FILM_PRUNE_FLOOR.json`
  receipt. Local disk held source only. The ~1.8 GB camera rasters are never materialised: the seg
  leg renders in 10-pair chunks and scores them in place, so there is no raster payload to discard.
- Every heavy leg went through `tools/launch_detached_process.py` with an armed done-receipt and
  derived resource budgets; every wait was a background receipt-bound until-loop, never a
  foreground clock. No `ScheduleWakeup`.
- CPU only. MPS was not used at all — nothing here needed a gradient.
- Lane `ddm_rq1_renderer_coarsening_then_field_reabsorption_20260912`, claimed.
- Provenance pins: HEAD `14cc70a63`; charter sha `5610197ab7548ee5…`; `ddm_ren1` memo
  `063b5dd1f5f97860…`; `ddm_ren2` memo `3b9cdbbdb6a7bfdf…`; `ddm_sj1` pass-6 charter
  `f2bd0979ba1e954e…` and memo `d57ab4d55b3d7b0d…`; multipass memo `3d848e593881b864…`; pass-3 memo
  `ab14594f9f91f939…`; move 48 seal
  `.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json`
  (archive `d830edd3…`, tree
  `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime`); field
  `a92e7d90…`.

### OPTIMAL FORM

Reference forms honoured: the shipped packer on the shipped float weights (no toy quantizer — the
control's byte identity proves it); `ddm_ren1`'s n600 seg instrument on the DALI GT lineage (the
authority the pricing instrument scores against); `ddm_ren2`'s member pricer, the first in this
campaign proved byte-identical to the shipped encoder. Declared deltas: the depth table /
`keep_percent` are the rung itself; variants (f)/(g) are a **SCOPE** addition (two more rungs to
measure the size axis rather than extrapolate it) and (h)/(i) are a **SCOPE** addition in the other
sign (they price `ddm_ntb2`'s named prize by measurement instead of citation). No MECHANISM was
reduced: every row is a real encode, a real parse-back, and a full n600 `d_seg` through R.

The frontier is unchanged: **composition S 0.13638261682704697 @ 179,111 B `[contest-CUDA T4 n600]`
(move 48)**.
