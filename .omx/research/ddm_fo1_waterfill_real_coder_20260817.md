---
arm: ddm_fo1
date: 2026-08-17
axis: "[macOS-CPU advisory]"
score_claim: false
promotable: false
verdict_scope: "formulation -- the M0-M8 coder family on sr1's 41-cell waterfilled support of rt1's free label boundary at n600"
title: "sr1's waterfilled seg-correction channel SURVIVES a real coder: 4,308 B round-trip-verified against a pre-registered 5,066 B bar (14.9% under), only 0.74% above the ideal entropy sr1 could not price -- the pose-null seg-edit family gets its first working supplier, and the binding uncertainty moves off rate onto eta"
---

# ddm_fo1 — the real coder on sr1's waterfilled support

## ANSWER FIRST

**The channel clears the bar. 4,308 B against 5,066 B — 756 B (14.9%) under.**

pn2 §7 pre-registered the bar before any run: real coded bytes ≤ 5,066 B → the waterfilled
channel is a real supplier; > 5,066 B → sr1's ΔS is an ideal-entropy artifact and rt1's CLOSED
verdict is restored. I raced 9 mask coders and 3 target-class coders on sr1's exact 41-cell
support, verified every payload by decoding it back through the same online context machine, and
landed:

| | bytes |
|---|---:|
| pre-registered bar (frozen) | **5,066** |
| real total, **pre-registered M0–M7 only** | **4,310** |
| real total, incl. one added coder (M8) | **4,308** |
| sr1's IDEAL entropy limit | 4,276.17 |

**The verdict does not lean on my addition** — the pre-registered set alone clears the bar by
756 B. The real coder came in **0.74% above the ideal**, against **18.47%** of available headroom.
sr1's supplier claim was carrying a 25× safety margin it never knew it had.

**Routing consequence: the pose-null seg-edit channel composes.** rt1's CLOSED verdict stands for
the *describe-everything* framing it was drawn on and is **not** restored for the waterfilled
sub-support. The binding uncertainty on this family is no longer the rate leg — it is η, the
realization efficiency, still measured at n=12 seeded-random pairs.

## §1 What was owed, and what the number rests on

sr1's `SR1_WATERFILL.json` says of itself: *"IDEAL conditional-entropy limit: empirical per-cell
H(p), NO model cost and NO coder inefficiency. This is a CEILING on the family, not a coder
result."* pn2 §6 named the gap plainly — *"The headroom is measured, the coder is not"* — and
sealed FO-1 to close it. This memo is that row.

The bar's arithmetic, reused not re-derived: `η_projected × 6,512 flips × 1.273108 B/flip`, with
pn2's pooled n=12 **projected η = 0.6111** → 5,065.6 B. (My charter quoted η 0.6224 from an
earlier draft of pn2; pn2's final n=12 memo carries 0.6111, which is what 5,066 B is built from.
The bar value is unchanged and I froze it.)

## §2 The selection reconstructs EXACTLY — the premise survives its own control

The first thing that could have failed is the premise, not the coder: if any cell factor were not
receiver-derivable, "free selection" breaks and the channel owes bytes it never budgeted.

I rebuilt the 41-cell selection from the transmitted labels alone using sr1's `cell_definition`
(own class × lowest differing 4-neighbour class × min(degree, 4) × row band — 1,200 cells), then
checked it against sr1's retained payloads rather than against sr1's prose:

| control | result |
|---|---|
| `cell_band_px` vs sr1 retained | **byte-identical** (sha `739d33c27e739416…`, sum 2,551,464) |
| `cell_flip_px` vs sr1 retained | **byte-identical** (sha `2d0b71464d3901e3…`, sum 34,666) |
| target-class rate | `0.22530701479359683` bits/flip — sr1's value to 17 decimals |
| cells / flips / ideal bytes | **41 / 6,512 / 4,276.171156196116 B** vs sr1's 41 / 6,512 / 4,276.171156196069 B |
| rt1 `free_band_mask` == boundary(labels), every frame | verified, fail-closed |
| rt1 `flip_mask_vs_gt` == (argmax_base != gt), every frame | verified, fail-closed |

Every **per-pixel** factor is label-derived, so evaluating membership is free. **One thing is
not**: the *set* of 41 cell ids was chosen using GT-derived flip densities, which makes it
this-clip side information and therefore COUNTED. sr1 excluded it; the bar excludes it; I priced
it anyway — **52.4 B** as a 41 × log2(1200) index list (150 B as a 1,200-cell bitmap). **At the
cheaper price the total is 4,360 B, still under the bar.** This is the honest answer to my
charter's "if any factor turns out NOT receiver-derivable" clause: the per-pixel premise holds
exactly, the fixed table does not, and the fixed table is small enough not to matter.

## §3 The race — 12 coders, every payload inverted, nothing asserted

Support: **94,124** band pixels (3.689% of the 2,551,464-pixel band) carrying **6,512** of 34,666
band flips (18.785%). Density **6.9185%** against the band's 1.3587% — **5.09× denser**.

| tag | bytes | b/flip | prereg | coder |
|---|---:|---:|:--:|---|
| M8 | **4,123** | 5.065 | no | CABAC full-band-walk order, support-only symbols (pair × run × temporal) |
| M7 | **4,125** | 5.068 | yes | CABAC support-walk (pair × run × temporal), 88 contexts |
| M5 | 4,140 | 5.086 | yes | CABAC raster (pair × causal-neighbours × temporal) |
| M6 | 4,230 | 5.197 | yes | CABAC support-walk (run × temporal) |
| M3 | 4,274 | 5.251 | yes | static binary AC (i.i.d. realized) |
| M4 | 4,281 | 5.259 | yes | adaptive binary AC, order-0 |
| M1 | 4,354 | 5.349 | yes | brotli(packed) q11 |
| M2 | 5,288 | 6.496 | yes | lzma(packed) preset 9\|EXTREME |
| M0 | 11,766 | 14.455 | yes | raw packed bits (no coder) |

Target class, coded for real because sr1's 4,276 B includes a 0.2253 bit/flip target term:

| tag | bytes | b/flip | coder |
|---|---:|---:|---|
| T2 | **185** | 0.227 | adaptive binary-tree AC, context = (own, partner) — both free |
| T1 | 1,386 | 1.703 | adaptive binary-tree AC, order-0 |
| T0 | 2,442 | 3.000 | raw 3 bits/flip |

**Every one of the 12 payloads round-trips.** The flags are earned, not asserted: each arithmetic
payload is decoded back through the same function that encoded it and compared against the truth
field. The proof is auditable from disk without trusting any flag —
`roundtrip_decoded_mask_best.npy` and `restricted_mask_bits.npy` carry the **same sha256**
(`8bca66ec89eb830a…`). A full n600 rerun reproduced **all 12 payloads byte-identically**.

M8 is an addition beyond pn2's pre-registered M0–M7, disclosed as such: it walks the full label
boundary and codes only support symbols, keeping the curve geometry a sparse sub-support breaks.
It bought **2 bytes**. Extra coders can only lower the total, which is the conservative direction
for a ">bar" verdict; the ">bar" verdict did not happen and the pre-registered set clears the bar
on its own, so M8 is decoration either way.

## §4 Why the real coder got so close to an ideal that assumed no model cost

The ideal credited itself 41 per-cell probabilities for free. A real coder must either transmit
them or learn them. **It learned them, and the learning cost was 30 B.**

- Ideal **mask-only** entropy 4,092.77 B; M7 4,125 B (**+0.79%**), M8 4,123 B (+0.74%).
- Ideal **target** term 183.40 B; T2 185 B (**+0.87%**).

The mechanism is that the CABAC contexts (edge-pair × run-state × temporal) **re-derive** the
per-cell conditioning from label geometry the receiver already has, so the model never has to be
sent. The evidence is in the spread: the static i.i.d. coder M3 lands at 4,274 B — 4.4% above the
per-cell ideal, exactly the value of the conditioning — and the context coders recover almost all
of that 4.4% for free. The 6,512 symbols are also enough for an adaptive coder to amortize
start-up; my own 20-frame smoke, where they are not, ran **71.7%** above ideal.

There is a second, larger reason the waterfill wins that has nothing to do with coder quality:
**density buys bits.** rt1's full-band M7 spends 7.447 bits per flip; on this 5.09× denser support
the same coder spends **5.068** — **31.9% fewer bits per flip**, because H(p)/p falls as p rises.
That is the waterfill's whole thesis, and it now has a measured coder behind it.

Where the density lives, for the next arm: **84.3% of the described flips sit in row band 3**
(rows 144–191, the horizon), and the single largest cell is Lane-owned against a Road partner
(2,736 flips, 42% of the total). By own class: Lane 50.4%, Road 30.4%, Undrivable 10.4%,
Movable 6.4%, MyCar 2.5% — MyCar cells are the densest at 31.7% but carry almost nothing.

## §5 The joint arithmetic on real bytes

| η | net ΔS, **real coder** (4,308 B) | net ΔS, sr1 ideal |
|---|---:|---:|
| 0.6235 (sr1's selection η) | **−0.000573** | −0.000595 |
| 0.6111 (pn2 projected — the bar's η) | **−0.000505** | −0.000526 |
| 0.5651 (pn2 **unprojected**) | **−0.000251** | — |
| 1.0000 | −0.002652 | — |

**96.4%** of sr1's headline ΔS survives the real coder. Break-even η on real bytes is **0.5196** —
below pn2's unprojected pooled η (0.5651) as well as its projected one, so on this support the
seg leg supplies **even without** the pose-null projection. (The pose leg is a different matter:
pn2's finding is that the projection removes the pose tax, and that leg still needs it.)

At η = 0.6111 the channel closes **5.26%** of the 0.0095973 S gap between hv1's 0.15959729 base
and 0.15.

## §6 What this does and does not settle

**Settles:** the rate leg. sr1's 4,276 B was not an artifact — a real, causal, round-trip-verified
coder reaches 4,308 B on the same object, and the family had 18.47% of headroom where it needed
0.74%. rt1's CLOSED verdict is **not** restored; it was drawn on the describe-everything framing,
which pn2 §5 independently reconfirms as a NON-SUPPLIER (+0.004171 S even with the pose leg
zeroed). Waterfilling is what separates the two, and it is now priced.

**Does not settle:**
- **Not a score.** `[macOS-CPU advisory]`, `score_claim=false`. No archive was built, no
  `upstream/evaluate.py` ran. The pointer did not move and this arm did not move it.
- **η is the binding uncertainty now.** −0.000505 S is entirely conditional on η = 0.6111, which
  is n=12 seeded-random from pn2, not a LIVE n600 verdict (m96: a random subset may refute a bar,
  it may not license a live one). The channel's whole margin lives inside that number.
- **The realization is unbuilt.** This prices *naming* 6,512 flips and their target classes. It
  does not build the decoder-side edit, and η is precisely the measure of how much of a named flip
  a realization actually recovers.
- **−0.000505 S is 5.26% of the gap.** A real supplier, and a small one. It composes; it does not
  arrive.
- **verdict_scope: formulation.** A better coder on this same support could go lower — nothing
  here bounds the family, and the 41-cell selection itself was optimized against an ideal-entropy
  objective, not against these realized bytes. Re-running the waterfill with **measured** per-cell
  coder bytes is the obvious next refinement and would move the selection, probably outward.

## §7 The one thing I would tell the next arm

The cell-set side info (52.4 B) is the only counted item the waterfill framing hides, and it is
also the item that grows if a future arm waterfills harder. At 41 cells it is 1.2% of the payload
and irrelevant. At 300 cells it would be ~385 B and would start to eat a margin that is 756 B
wide. Price it inside the waterfill's inclusion test, not after it.

## §8 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_fo1_waterfill_real_coder/` — full sha manifest in
`RECEIPT.md`. Every coder payload (9 mask + 3 target), the restricted mask, the support index and
frame offsets, the target values, the selected cell ids, the reconstructed histograms, the
round-trip-decoded mask, both run receipts, both determinism-repeat receipts, `PROGRESS.jsonl`,
and both run logs. The repeat's duplicate blobs were removed only after their byte-identity was
recorded by sha256 in `FO1_CODER_RACE.repeat.json`.

Tool landed this unit: `experiments/ddm_fo1_waterfill_real_coder.py` — reuses rt1's range coder
and geometry verbatim so the bytes are comparable to rt1's race, and sr1's `cell_features`
verbatim so the support is the same object.

Consumed unmodified: rt1's `argmax_base` / `flip_mask_vs_gt` / `free_band_mask` /
`flip_target_class`, sr1's `cell_band_px` / `cell_flip_px` / `SR1_WATERFILL.json`, hv1's
`decoded_spatial_tokens.rc64.bin` (ep0634), qs3's `gt_argmax_n600.npy`. **`upstream/` was neither
read nor written by this arm.**
