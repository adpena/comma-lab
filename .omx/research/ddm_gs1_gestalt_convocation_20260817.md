---
schema: ddm_gs1_gestalt_convocation.v1
date_utc: 2026-08-17
arm: ddm_gs1 (23rd convocation — adversarial full-corpus pass on the gestalt)
lane_id: "lane_ddm_gs1_gestalt_convocation_20260817"
convocation: 23rd (operator-convened; Schmidhuber LEAD; T3)
council_tier: T3
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
modal_dispatched: false
scorer_run: false
spend_usd: 0
axis: "[local-CPU $0 re-derivation over MEASURED receipts] — NEVER a score"
verdict_scope_default: "per-claim, stated inline"
own_vehicle_frontier: "rr4 S 0.15853325034789678 @ 181,161 B [contest-CUDA T4 n600] — UNMOVED by this unit"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_gs1 — the gestalt is REFUTED at its core claim: 68.30% of the fortnight's gain was descent

## §0 ANSWER FIRST

**The gestalt's headline claim P1 — "Every pointer move was an operator, none was descent" — is
REFUTED at the primary receipts.** The fortnight's four contest-CUDA pointer moves total
ΔS −0.00342188793034498. Of that, **−0.00233716492545882 (68.30%) came from gradient training**:
a 480-epoch QAT burn (6.49 h wall) and its 154-epoch continuation. The gestalt memo names three
moves and **omits the largest one entirely** — the move that alone carries 53.84% of the gain.

**The supporting premise is worse than incomplete: it cites a fit that a HIGH-severity CONFIRMED
finding had already voided two days earlier.** The draft says "the e960 burn ASYMPTOTED (~480 B
remaining) — descent saturates." The same lineage's own refit says the opposite, in its own words:
**"the descent has NOT floored"** — power-law tail, ~1.2 KB per doubling of QAT epochs. And the
"~480 B" is not headroom; it is **the 482 B by which the run had already beaten the asymptote the
draft is quoting** (`av1` F3, HIGH, CONFIRMED).

**PP3, the "falsifiable, pre-registered prediction test," had already fired against itself before
it was written.** Its named falsifier is "if a plain longer burn moves the pointer next, this
gestalt is WRONG in a named way." A plain longer burn moved the pointer on 08-15 **and** 08-16 —
two and three days before the sentence was typed. PP3 is post-diction, and leg (b) is `ns1`'s
prior-day finding reproduced verbatim in substance.

**What survives.** The gestalt's ONE-SENTENCE form — "training's remaining role is to produce
better frozen objects for the operators to act on" — is defensible and largely CONFIRMED. The
sharp object-level claim built on top of it is not. Five claims CONFIRMED, four REFINED, four
REFUTED.

**Pointer: UNMOVED. A convocation should not move it, and this one did not.** No launch, no Modal,
no scorer run, $0.

---

## §1 THE MEASURED CHAIN (re-derived here, not quoted)

Every S below is recomputed from components at 28-digit precision. Source:
`.omx/state/continual_learning_posterior.json` `accepted_anchor_history` +
`.omx/state/canonical_frontier_pointer.json`.

| # | date | row | S | archive B | ΔS | share | mechanism |
|---|---|---|---:|---:|---:|---:|---|
| 0 | 08-10 | cp135 intake (base) | 0.16195513827824176 | — | — | — | base |
| 1 | 08-14 | MC36 Variant C | 0.16193445788044480 | 186,269 | −2.068040e-5 | **0.60%** | EDIT (post-hoc, in-compile compensation) |
| 2 | 08-15 | **e480b v2** | 0.16009202615715580 | 183,502 | **−1.842432e-3** | **53.84%** | **480-epoch QAT training burn** |
| 3 | 08-15 | hv1 ep0634 | 0.15959729295498598 | 182,759 | −4.947332e-4 | **14.46%** | **+154 more epochs**, harvested by selector |
| 4 | 08-17 | rr4 recode | 0.15853325034789678 | 181,161 | −1.064043e-3 | **31.10%** | RECODE (HPAC context, lossless) |

**Descent-sourced (rows 2+3): 68.30%. Pure recode (row 4): 31.10%. Post-hoc edit (row 1): 0.60%.**

### 1.1 The finding the draft missed entirely — distortion has been frozen since MC36

Implied distortion `S − 25·B/37,545,489` across the chain:

```
MC36        0.037905576541331065
e480b v2    0.037905576541331113
hv1 ep0634  0.037905576541331066
rr4         0.037905576541331096
```

Identical to **14 significant figures**; the residue is float round-off in the 17th digit. `hv1`
states the mechanism outright: *"d_seg contribution 0.029611 → 0.029611 (inherited) … valid because
the decoded frames are byte-identical."* `rr4`'s parse-back closes it:
`decoded_field_bit_identical: true`.

**So three of the four moves changed only RATE, and the decoded token field has been bit-identical
since 08-14.** Only MC36 touched the field: it bought exactly **−3.2000e-5** of distortion for
**+17 B** of rate (net −2.068e-5). That is 0.60% of the fortnight and **0.375% of the remaining gap**.

This is a sharper and more useful statement than the draft's flat five-generator list, and the draft
does not contain it.

---

## §2 PER-CLAIM VERDICTS

| claim | verdict | receipt |
|---|---|---|
| **P1** every pointer move an operator, none descent | **REFUTED** | 68.30% descent-sourced (§1); enumeration omits the largest move; "saturates" premise voided by `av1` F3 |
| **P2** field independently groks the same thing (opal) | **REFINED** | Convergence real and confirmed at source, but we are **ahead** of opal by 6.163e-4; "near exhaustion on both sides" is unverified and half-refuted |
| **P3** zero-counted-byte mechanisms dominate | **CONFIRMED** | rule-118 lived; `cv1`: "the selector is 535 B and IS the decode program" |
| **P4** no single axis closes sub-0.15 | **REFUTED as written** | Self-contradictory in adjacent clauses; correct claim is *pose alone* cannot |
| **P5** realized-vs-modeled is THE epistemic split | **CONFIRMED** | Strongest claim in the draft; `rr2` staging infidelity + `qs4` stale compensation are new instances |
| **P6** knowledge decay outpaces generation | **CONFIRMED — and self-instantiating** | The gestalt memo commits the error **three times** (§3.4) |
| **P7** determinization compounds | **CONFIRMED** | `rr4`'s landing cure was firing `candidate_runtime` UNMODIFIED via the canonical fire tool |
| **P8** pre-registered falsifiers convert ambiguity to cheap verdicts | **CONFIRMED** | `tc1`, `oq1`, `rr4`'s three falsifiers hitting exactly |
| **PP1** campaign exhibits grokking phenomenology | **REFINED** | Narrative fit; the measured trigger includes a burn that had NOT floored, not only a representation change |
| **PP2** one-object doctrine, measured | **REFINED** | `hg1` MEASURED the actual algebra: commutative idempotent join-semilattice; value function **not submodular** ⇒ no greedy guarantee. The draft's "search over an operator algebra" understates a known constraint |
| **PP3** the prediction test | **REFUTED (leakage + already-fired falsifier)** | §4 |
| **R1** alive subalgebra = {lossless recode, select, constrained-solve edit} | **REFINED** | `qs5` IS a genuine Schur constrained solve (verified, not a relabel), but the 07-18 law is witness-vehicle-scoped, one leg is an operator-caught STRAWMAN, and `gx1` measured the closure SCALE-CONDITIONAL |
| **R2** the pattern predates the fortnight (FEED-gy) | **REFUTED in attribution** | FEED-gy ranked CAPACITY "THE #1 suspect" and made it "THE decisive plateau test" — that CONTRADICTS the gestalt. FEED-gz (same day) is the correct receipt |

**Counts: 5 CONFIRMED · 4 REFINED · 4 REFUTED.**

---

## §3 COUNTER-EVIDENCE (hunted with equal energy, per charter)

### 3.1 (a) DESCENT WINS — the claim breaks here

`ddm_wc2_hpac_mps_port_20260814.md:511-517` — the e480b endpoint: *"ep480 terminal joint estimate
**131,220 B** … **−3,103 B vs the 60-ep endpoint** (134,323), and **below the fitted 60-ep QAT
asymptote** (132,798). Wall 23,372.5 s (6.49 h)."*

`:631-636` — *"**THE ROW LANDED.** … **S = 0.1600920261571558 @ 183,502 B [contest-CUDA T4 n600]**
… Δ vs the prior frontier … **−1.8424e-3** (~526 canonical bands) — the largest single pointer move
of the micro-campaign era."*

`:653-657` — *"The e480b endpoint refit … **REVERSED the law form**: power (alpha 0.14) beats
exp-floor at this depth, fitted QAT asymptote 118,147 B — **the descent has NOT floored.** Projected
~1.2 KB/doubling of QAT epochs (~−8e-4 S per doubling)."*

`:674-677` — the route ranking puts *"(a) **THIS e960 QAT token burn** (~1.2 KB/doubling ≈ −8e-4 S)"*
**first**, ahead of every operator route.

`:694-696` — *"First forward row ep482: joint est **131,743 B** — already 1,055 B below the old QAT
exp-floor 132,798 (the §5f power-law refit's 'not floored' call **confirmed at first contact**)."*

**The "saturation" premise is a stale-fit artifact.** `ddm_av1_wd2_earlystop_adversarial_review_20260815.md`
F3 (**HIGH, CONFIRMED**): *"A non-monotone noisy byte trace is declared asymptotic because the fit
floor is pinned to the current minimum; future checkpoints are then discarded even though the same
run later beats that 'floor.' Fit receipt SHA 9918be52…: y_inf=130875 … selector later chooses
130,393 B."* `130,875 − 130,393 = 482 B` — the draft's "~480 B remaining" is **exactly the margin by
which the run had already beaten the floor it is citing as evidence of saturation.**

Advisory sisters (not pointer moves, correctly labelled): `ef3000` crossed below init for the first
time in ten runs (−2,286 flips, `[macOS-MPS training-signal]`), `ef6000` reproduced it (−2,755 at
interior step 5,200), and `cw1` measured that even that endpoint stopped with **0.64% of its
integrated LR budget left** — *"A run that stops improving with 0.64% of its learning rate left has
not demonstrated a floor. It has demonstrated that it stopped when the schedule stopped."*

### 3.2 (b) STATIC CODER RACE WINS — the claim SURVIVES

Every static coder race in the window LOST. `mz1`: 8/8 full-section alternatives lost, exact savings
**0 B**. `dc1`: token +5 B, semantic/carrier byte-identical to shipped — *"Racing them is a
definitional no-op."* `me1` §5 on the live rr4 law: **+67.12, +359.47, +561.21, +662.83, +689.11 B —
all negative**, with a theorem (*"Averaging cannot beat its best member"*). One honest footnote:
`rc64p` (08-10) beat lc2 by **4 bytes** — ~400× below the naming bar, on a superseded lineage, and
the contest-CPU pointer never moved. It does not break the claim.

### 3.3 (c) POST-HOC VALUE CORRECTION SUCCEEDS — SURVIVES on the "arbitrary" qualifier

One genuine dent: `qs2` — a **stored** compensation object recoded at 5.67 B/pair (+34 B), applied
post-hoc, **ADMITTED** at net realized ΔS **−4.374914e-6** on a real T4 dual-axis component
instrument, n600. It is sub-band (|−4.37e-6| < ±3.5e-6/side), so it is not a pointer move — but it
is a stored post-hoc correction that produced a measured real-chain win, and it later **shipped**
inside MC36. The failures balance it: `qs4` **+2.438e-4 REFUSED** (stale compensation), `qs5`
**+2.520e-6 REFUSED**.

**The irony worth handing back:** `rv2`'s two hardest closures both say the *only* way to reopen the
dead post-hoc axes is **joint descent**. The corpus's own escape hatch from operator exhaustion is
training — which is the thing P1 says has saturated.

### 3.4 P6 is self-instantiating — the gestalt memo commits its own diagnosed error three times

1. **P1** cites the e960 asymptote as evidence of saturation. That fit was voided by `av1` F3
   (HIGH, CONFIRMED) two days earlier, and by the e480b refit's *"has NOT floored"* three days earlier.
2. **P4** states "NO single axis closes sub-0.15; composition is mandatory" and then, **in the very
   next sentence**, gives two single-axis closures ("Seg −28.8% or rate −12,815 B"). Headline
   contradicts body — the exact detector class the corpus tracks.
3. **R1** presents the 07-18 post-hoc law as flat-dead without citing `gx1` (08-16), which measured
   the closure **SCALE-CONDITIONAL**: *"the pose screen is LINEAR in bytes saved, so 'post-hoc weight
   edits are dead' is SCALE-CONDITIONAL, not a clean family kill"* — the 139× miss falls to **1.7×**
   at the full drop pool.

---

## §4 PP3 — LEAKAGE VERDICT: LEAKED, AND THE FALSIFIER HAD ALREADY FIRED

**Leg (b) is not a pre-registration.** `ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md`
— written the **day before** the gestalt — states as its ranked finding: *"**P1 — Train-for-editability**
(the inverse of every §B refusal)."* PP3(b) reproduces it. Post-diction, not pre-registration.

**Leg (a) conflates two mechanisms, and the half the draft names as the prediction has already been
measured negative.** PP3(a) predicts the next rate win comes from "an opal-class online-adaptive
prior on OUR stream (me1 stage-7 leg iii)". `me1` §5 read our shipped law at source and found
**"our shipped rr4 law is already opal-class"** — the transport form is identical; the only
difference is 1 joint context of 51,200 bins vs opal's 55 mixed families. `me1` tested that
difference directly on full n600, decode-identical: **all five alternatives LOST** (+67 to +689 B).
The **static context-mixture** half of opal-class is therefore measured DEAD on the live object. The
**online-adaptive** half (state regenerated from the decoded prefix) is genuinely unmeasured. PP3(a)
survives **only** on the narrow online-adaptive leg, and the draft does not make that distinction.

**Leg (c) restates `rv2`'s existing closure** ("pose CLOSED as a post-hoc axis; reopening requires
joint descent"). Not new.

**And the falsifier had already fired.** PP3's own text: *"If instead a plain longer burn … moves the
pointer next, this gestalt is WRONG in a named way."* A plain longer burn moved the pointer on 08-15
(−1.842e-3) and again on 08-16 (−4.947e-4). **The named condition was already satisfied when the
prediction was written.** By its own terms, the gestalt is wrong in the way it named.

---

## §5 SURVIVORSHIP-CORRECTED P1

**First, a charter correction.** The charter names the probe-outcomes ledger (662 rows / 23 callers)
as the survivorship adjudicator. Measured: it holds **728 rows, 436 unique probe_ids, 279 substrates**,
and **601 of 728 rows (82.6%) are May 2026**; only **66 rows are August**. The store is temporally
mismatched to the claim and **cannot** adjudicate the fortnight. I built the denominator from the
acceptance oracle instead, which is the exact arbiter: every exact-gate fire is one attempt.

Exact-gate scored candidates in the window (`.omx/state/modal_call_id_ledger.jsonl`, deduplicated
across retry storms):

| class | candidates fired | pointer moves | hit rate |
|---|---:|---:|---:|
| EDIT / constrained-solve (`qs1,qs2,qs4,qs5,re1,mc36,mt1`) | 7 | 1 | **14.3%** |
| SELECT / re-prior from a burn (`rx2 e480b`, `hv1 ep0634`) | 2 | 2 | **100%** |
| RECODE (`rr2`, `rr4`) | 2 | 1 | **50%** |
| **DESCENT fired at the oracle directly** | **0** | **0** | **undefined** |

**The honest correction has two parts, and they point opposite ways.**

1. **P1's factual enumeration is survivorship-clean in the narrow sense** — of 11 exact-gate
   candidates, 4 moved the pointer, and all 4 were *submitted through* an operator (build → gate).
   Overall hit rate **4/11 = 36.4%**. An algebra whose generators refuse ~64% of the time is a
   weaker claim than "operators don't saturate."

2. **P1's causal gloss is unsupported, and in the strongest available reading it is backwards.**
   Descent never fired the acceptance oracle directly — it has **zero attempts** in the denominator.
   You cannot conclude "descent saturates, operators don't" from a record in which descent never
   fired the gate. But descent is not merely un-tested: it **supplied the value that two of the four
   winning operators harvested.** `ep634` did not exist until 154 epochs past `ep480` were run; the
   `e480b` prior did not exist until a 6.49-hour burn produced it. The SELECT class's perfect 2/2 is
   *the measure of what the burn produced*, not evidence that selection beats training.

**Corrected P1: every pointer move was HARVESTED through an operator at the acceptance oracle, and
68.30% of the harvested value was MANUFACTURED by gradient descent. Operators are the withdrawal
mechanism; the burn is the deposit.**

---

## §6 R1 VERIFICATION AT THE 07-18 LAW'S OWN RECEIPTS

**`qs5` is a genuine constrained solve — CONFIRMED, not a relabel.**
`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md:16-20`: *"the in-compile compensation is PROVEN
(MEASURED). d_pose landed BELOW base: the **exact-object Schur solve** not only cancelled frame-1
pose leakage, it slightly improved it."* That is the 07-18 law's "constrained terminal SOLVE"
branch, executed. R1's mapping holds.

**But R1 overstates the law in three ways the law itself declares:**

1. **Scope.** The law's own text binds it to *"the level-set **witness** vehicle."* The current
   frontier is the PR135/rr4 lineage — a different vehicle. Transferring it is exactly the
   cross-regime constant-transfer genus the corpus bans.
2. **One of three legs is a strawman, by the operator's own catch.** *"CAVEAT: the phase leg is a
   STRAWMAN, not a clean confirmation (operator caught 2026-07-18) … the +56.6%→+161% 'coupling wall'
   is SUBSTANTIALLY a strawman artifact, NOT proof post-hoc-storage-of-phase is intrinsically dead."*
   R1 cites the law without the caveat.
3. **`gx1` measured it SCALE-CONDITIONAL** on 08-16 (§3.4 item 3). R1, written 08-17, does not cite it.

**And `qs5`'s own verdict was REFUSED (+2.519822e-6).** The constrained-solve generator's first
firing lost. It produced a portable *asset* (proven in-compile compensation) that MC36 later
consumed. Generators that refuse can still be productive — but that is a different claim from
"alive."

---

## §7 THE OPERATIONAL RE-RANKING — the draft's inversion is itself inverted

The draft concludes: *"me1 (the operator-composition engine) is the campaign's center of gravity."*
**The measured record does not support that ranking.**

**The closing arithmetic.** Gap to 0.15 = **0.00853325034789678** = **12,815.4 B** to remove
(181,161 → 168,346 B, seg+pose held).

- **Descent at its measured tail**: ~1.2 KB per doubling of QAT epochs ⇒ **10.7 doublings** ⇒
  480 → **~7.9e5 epochs**. Descent is NOT floored, but at its measured slope it is a **contributing**
  lever, not a closing one.
- **Operators on the frozen object**: `ns1` measured all three axes at floors within this regime;
  `me1`'s four coder architectures all lost; `mz1`'s 8/8 lost; the admitted bank covers **0.058%** of
  the gap (`gx1`).

**Neither class closes the gap alone. The draft is right that composition is mandatory — and P4
states that conclusion with the wrong supporting sentence.**

| rank | route | why (measured) | change vs draft |
|---|---|---|---|
| **1** | **e960 / e1920 QAT continuation + selector harvest** | Only route with a live measured slope (~−8e-4 S/doubling); supplied 68.30% of the fortnight; `wc2`'s own route ranking puts it first; "has NOT floored" | **PROMOTED** (draft ranks burns last) |
| **2** | **online-adaptive prior** (the *narrow* opal leg) | Static context-mixture half already measured dead by `me1` §5; adaptive-state half genuinely unmeasured; opal's claim sits 6.16e-4 **behind** us | **NARROWED** from PP3(a) |
| **3** | ce1/cw1 aligned-objective long burn | `ef3000`/`ef6000` first below-init descent in 10 runs, reproduced; but `[macOS-MPS advisory]`, never byte-closed | held; **owed a byte-closed row** |
| **4** | me1 micro-edit composition | Real but small: banked qs2+re1 = −5.58e-6 ≈ **0.065%** of the gap, and both need RECOMPILE against the new coder | **DEMOTED** from "center of gravity" |
| **5** | pz4/js8 pose line | Ceiling −0.0083 < gap 0.00853; cannot close alone (P4's *correct* content) | held |

**The composition hazard the draft does not carry.** `rr4`'s own receipt: *"the banked micro-edit
offsets (qs2 −4.375e-6, re1 −1.207e-6) were compiled against the OLD coder — they need RECOMPILE
against the new stream before any union fire."* And `cp1` MEASURED a union that looked safest and was
a **net loss** (+3.469e-05: the fold refunded −4.461e-05 of rate and spent +7.931e-05 of pose).
`hg1` gives the formal reason: the drop algebra is a **commutative idempotent join-semilattice** —
structurally flat, every commutator zero — but its **value function is neither modular nor monotone,
so it is not submodular and greedy waterfill carries no approximation guarantee.**

**Operational consequence: never fire a union on summed credits. Re-measure the union.**

---

## §8 THE ONE NEW FALSIFIABLE PREDICTION THE CORPUS LICENSES

The draft's PP3(a) predicts the next rate win comes from an operator, not a burn. The corpus
licenses the **inverse**, and it is sharper:

> **GS1-PRED.** The next pointer move of magnitude ≥1e-4 will come from a **NEW CHECKPOINT** — the
> e960/e1920 QAT continuation harvested by the selector — and **not** from a post-hoc operator
> applied to the frozen rr4 object.
>
> **Grounds (all measured):** the QAT descent law was refit to a power tail with fitted asymptote
> 118,147 B and is explicitly *"NOT floored"*; ep482's first forward row already sat 1,055 B below
> the previous exp-floor; the burn supplied 68.30% of the fortnight; and every operator family on
> the frozen object has now measured at a floor (`me1` 5/5 negative, `mz1` 8/8 negative, banked bank
> = 0.058% of the gap).
>
> **Falsifier (pre-registered, symmetric):** if the next ≥1e-4 move is produced by an operator on the
> frozen rr4 object with **no new checkpoint** in its lineage, GS1-PRED is wrong and the draft's
> PP3(a) is right.

This is a real bet with a named loser. It is the inverse of the memo it reviews.

---

## §9 COUNCIL — DISSENT VERBATIM

**Schmidhuber (LEAD).** "Compression is understanding, and the record says the compressor that
learned beat the compressor that was edited by 68 to 1. I accept the refutation of P1. But I dissent
from any reading that says operators are unimportant: the burn's value was invisible until a selector
found ep634 among 81 checkpoints. Manufacture and withdrawal are both necessary. The draft's error is
not preferring operators — it is *narrating* the deposit as a withdrawal."

**Contrarian.** "I dissent from the confidence of GS1-PRED. Ten-point-seven doublings is a number
that should frighten us. The e960 continuation buys ~1.2 KB — about 9% of the gap — and then the next
doubling costs twice the wall-clock for the same 1.2 KB. Ranking it #1 is right for the *next* move
and wrong for the *campaign*. Do not let a live slope be mistaken for a sufficient one."

**Assumption-Adversary.** "The assumption every claim here operates within — including my own — is
that **archive bytes are the binding axis**. Four of four moves were rate. Distortion has not moved
in three days and moved 0.375% of the gap in the one move that touched it. We have collectively
stopped attacking seg because seg stopped yielding, and then we wrote a theory explaining why seg
does not need attacking. That is HARD-EARNED for rate and **CARGO-CULTED for seg**. `gc16`'s
actuator-class verdict (08-04) said the same thing and we did not act on it."

**Hotz.** "Four archives, one member changed. You have a 110 KB token stream and a 70 KB model
section, and you have been polishing the 110 KB. `wc2` says the 70 KB decomposes 13,619 HPAC +
34,763 semantic + 22,161 carrier and that 8/8 races lost — that means nobody has actually *attacked*
the 56,938 B of frozen semantic+carrier, they only re-compressed it. Attack the representation, not
the coder."

**Dykstra (CO-LEAD).** "The composition question is settled formally and it is bad news: not
submodular, no greedy guarantee. Every 'compose the bank' plan in the queue is unbacked until the
union is re-measured. `cp1` already paid for that lesson once."

**Rudin (CO-LEAD).** "The gestalt memo is an explanation that does not survive its own receipts. I
note approvingly that its failure mode is exactly the one it diagnosed in P6. An honest theory that
falsifies itself in three places is more useful than a vague one that cannot."

---

## §10 STORES CONSULTED

Graph-memory recall (38,277 nodes / 157,950 edges), queries beyond MAIN's two: *training breakthrough
descent win pointer moved* · *coder race static entropy coder win* · *post-hoc correction sidecar
success measured win* — plus targeted lineage and counter-evidence sweeps over the fortnight.

Primary receipts re-derived, not quoted: `.omx/state/continual_learning_posterior.json`
(189 accepted anchors, 12 in window) · `.omx/state/canonical_frontier_pointer.json` ·
`.omx/state/modal_call_id_ledger.jsonl` (279 window rows) · `.omx/state/probe_outcomes.jsonl`
(728 rows, month histogram) · `ddm_au1_20260805/au1_corrections_index.jsonl` (11,840) +
`au1_headline_vs_body.jsonl` (8,157) · `ddm_wc2_hpac_mps_port_20260814.md` ·
`ddm_av1_wd2_earlystop_adversarial_review_20260815.md` (F1–F6) · `ddm_hv1_harvest_compose_ep508_20260815.md` ·
`ddm_hv1_pointer_move_and_wd2_advisory_chain_20260815.md` · `ddm_mc36_promotion_complete_s_verdict_20260814.md` ·
`ddm_rr4_t4_verdict_pointer_move_20260817.md` · `ddm_rv2_frontier_adversarial_review_r1_20260817.md` ·
`ddm_gx1_gap_closure_composition_table_20260816.md` · `ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md` ·
`ddm_me1_micro_edit_engine_20260817.md` · `ddm_cp1_composition_matrix_20260802.md` ·
`ddm_hg1_negatives_as_geometry_20260803.md` · `ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` ·
`ddm_qs2_r2_admitted_verdict_20260813.md` · `ddm_mz1_model_section_rate_race_20260815.md` ·
`ddm_dc1_decode_budget_conditional_coding_20260816.md` · `ddm_ef3000_first_descent_verdict_20260817.md` ·
`ddm_ef6000_double_window_verdict_20260817.md` · `ddm_cw1_corrected_window_20260817.md` ·
memory `post_hoc_stored_corrections_dead_joint_descent_required_law_20260718.md`.

Convocation lineage (gc1–gc21 + ph2/ph3/sc2/cv1, 23 ordinals; `gc3` verified never to have existed;
ordinal/label collisions documented at `ddm_gc17_retrieval_is_the_binding_constraint_20260731.md:34-39`).
PREDICTS receipts: `ph3:60` (min-entropy member selection at same distortion — the corrected R1
subalgebra, 17 days early) · `cv1:27-28` ("seg is CONSTANT to 7 dp … four consecutive pointer moves,
all pose or rate") · `gc16_full_stack_20260804:16-24` (actuator-class) · `gc19` ("not another broad
witness training run") · `gc14:23` (descent boundary-localized, OLS slope t = −0.09).
CONTRADICTS receipts: `FEED-gy` (`sub015_DAG_…_20260611.md:3914`, capacity = "THE #1 suspect";
`:3938` capacity probe = "THE decisive plateau test") — resolved against itself by `FEED-gz` `:3952`
("the plateau is NOT capacity") · `gc12:96-99` + `gc13:93` (from-birth burn = "the gap-sized lever") ·
`ph3 §10:251-253` (from-birth "can only beat the post-hoc bound") · `gc20` SL2.

**Frontier: rr4 S 0.15853325034789678 @ 181,161 B [contest-CUDA T4 n600] — UNMOVED by this unit.**
