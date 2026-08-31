# FORM vs FIT reconciled — bz2d's 2.178× and gf1's "one object" are the SAME law on TWO FIELDS, the generator packet is target-independent to 0.37%, and the four B-pairs close by recall

Date: 2026-08-31 · Author: MAIN · Cost: **$0** (two landed memos read at source + 5 lines of arithmetic)
Axis: no new measurement. `score_claim=false` · `promotable=false`
`verdict_scope`: **FORMULATION** — the HG1 four-stream analytic generator, on lb1's field. Inherits
gf1's own scope declaration verbatim; adds no scope it did not measure.

## Why this exists

Two arms landed on 2026-08-30 with headlines that read as opposites:

| memo | headline |
|---|---|
| `ddm_bz2d_distortion_verdict_20260830.md:147` | *"A successor body that wants born-small's rate should inherit the generator **FORM**, not the GT-fit field."* |
| `ddm_gf1_generator_form_capacity_verdict_20260830.md:1` | *"the mechanism is a **CAPACITY CEILING**"* … the ceiling is **TARGET-INDEPENDENT**, so **form and fit are one object** (task #1334) |

Naively: one says take the form without the fit; the other says you cannot separate them. Left
unreconciled, this is exactly the shape that spawns a wasted arm — "inherit the FORM onto lb1" is an
obvious next move, and it is already MEASURED and REFUSED. This is the [[m122]] recall discipline
applied to two of my own live memos before chartering anything.

## 1. THE RECONCILIATION — same law, two fields, and the packet barely moves

`bz2d` §7 held the FIELD fixed and swapped the REPRESENTATION. Its "bit-identical object" is
**bz2's own field**, not lb1's (`ddm_bz2d:120-134`):

| representation of **bz2's field** | bytes |
|---|---:|
| {RX1M hdr 14 + HPAC model 13,515 + coded tokens 90,529} | **104,058** |
| HG1C generator packet | **47,779** |
| | **2.178×** |

`gf1` fitted **the same generator form** to **lb1's field** (`ddm_gf1_generator_form_capacity_verdict:1`,
`ddm_gf1_capacity_gap_decomposition:204`):

| the same form on **lb1's field** | bytes |
|---|---:|
| generator packet | **47,603** |
| exact residual for a lossless round-trip | **385,448** |
| **total replacement** | **433,051** |

> ### THE JOIN NOBODY MADE: the generator PACKET is 47,779 B on one field and 47,603 B on the other — **0.37% apart**. What changes by 385,448 B is the **RESIDUAL**.

That IS gf1's target-independence, stated in bytes: the form costs ~47.7 kB to describe *whatever the
generator can produce*, on any target. bz2's field was produced by that generator, so its residual is
zero and the 2.178× is pure. lb1's field was not, so the residual is **8.10× the packet** and swamps it.

**Neither verdict is wrong and neither refutes the other.** "Inherit the FORM" is a design note for a
**successor body** — bz2d says so in its own words ("a successor body that wants…"). It is **not** a
transfer onto lb1, and the transfer has already been measured: **REFUSED at 5.09×.**

## 2. WHAT THIS CLOSES — the four B-pairs, by recall, at $0

This morning I recorded four unmeasured residue pairs (`ddm_rc_precheck_folded_never_fired:§5`):
**R+B · M+B · P+B · C+B** — keep lb1's renderer / HPAC model / pose / carrier, swap the token
representation to born-small's generator form. B was closed as a standalone body (`bo2`, 209×) but
never as a **pairing**, and [[m148]] says a closed leg can survive if another leg changes its object.

**That configuration is precisely what gf1 measured.** gf1 fitted the generator to lb1's field while
lb1's semantic renderer (30,856 B) and pose carrier (22,010 B) stay byte-identical — `bz2d:124-125`
proves those two sections are shared byte-for-byte across this lineage by substring hit at offsets 21
and 30,877 (shas `39d1be52…` / `932b979f…`). So R+B, M+B, P+B and C+B are **not four unmeasured
cells**; they are one measured configuration, at **5.09× over its own bar**.

**CLOSED at FORMULATION scope (HG1's four-stream analytic generator), with ONE exact reactivation
criterion, quoted from gf1 §4 verbatim:**

> *"any successor formulation with ≤0.109% capacity gap on lb1's field at a comparable packet."*

This does **not** close a structurally different generator. gf1 corrected its own scope from FAMILY to
FORMULATION the same day for exactly this reason — two targets is not two formulations.

## 3. THE BOUNDING ARITHMETIC — the identified levers reach ~2× of a required 10.30×

From `ddm_gf1_capacity_gap_decomposition` §8d, against gf1's own bar `packet_B + 0.2909 × mismatches < 85,020`:

```
budget headroom above the packet   37,417 B   =  128,624 mismatches allowed
mismatches present              1,325,033      ->  10.30x over
best available packet reduction        14 B    (0.03% of the headroom)
```

The packet axis is **spent** (§8d: *"There is no free rate win inside it"*) — the entire 10.30× must
come from mismatches. And the identified mismatch levers do not reach it:

| lever | measured reach | source |
|---|---:|---|
| free per-frame rigid horizon correction | **1.037×** (6.1% of displacement, 5.09→4.91×) | §2 |
| perfecting the dominant stream | **2.04×**, *and flagged optimistic in the stated direction* | §3 |
| composite-order DOF (all 24 swept) | **EXHAUSTED** | §6 |
| ordering / coder choice on the residual | already inside the 0.2909 B/correction price | §3a |

gf1 §4 states it plainly: *"a successor must beat HG1 by **10.3×** overall, and **no single-stream
perfection reaches it**."* The best identified case is **~2.04× of 10.30× — a 5.05× shortfall**, and
§3 says even that 2.04× is optimistic because the streams are not independent (Movable's 497,126 false
paints are counted against *other* classes' misses, so "perfect the horizon" silently also requires
Movable to stop over-painting).

## 4. Verdict

**The generator route is byte-cheap and reach-poor, and the two properties are decoupled: the packet
is target-independent, the residual is everything.** The 2.178× representational advantage is REAL and
is a fact about representing *generator-reachable* fields — it does not transfer to lb1's field, where
the same form leaves 1,325,033 mismatches.

This composes with the day's other closures rather than contradicting them:
[[the-cross-two-objects-each-hold-one-half-of-sub012]] and #1339's chasm say generators hold the byte
half and lb1 holds the distortion half; **this memo names the mechanism** — the generator's *reach*,
not its *rate*, is what fails, and the reach deficit is 10.30× against identified levers worth ~2×.

**verdict_scope: FORMULATION.** Reactivation is gf1's, unchanged and still one `count_nonzero`:
a successor formulation reaching **≤0.109%** capacity gap on lb1's field at a comparable packet.

## 5. Denominator

Candidates examined: **4** (the four B-pairs). Closed by recall + arithmetic: **4** — as ONE measured
configuration, not four. New exact quantities derived: **2** (the 0.37% packet target-independence
join · the ~2.04× vs 10.30× shortfall). Measurements run: **0**. Arms spawned: **0**. Arms PREVENTED:
**1** (an "inherit the FORM onto lb1" charter, already measured REFUSED at 5.09×). Dollars: **0**.

The exact pointer did not move. This unit did **not** achieve the goal — it reconciled two same-day
verdicts that read as a contradiction, closed a residue I had recorded as unmeasured, and stopped a
charter that recall already answers.

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
