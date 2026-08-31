# PER-PAIR ROUTING IS DEAD BECAUSE THERE IS NO TAIL — the generator's dominant error is ~1,489 px/frame in EVERY frame, and the address is only 75 B; the surviving cell is SPATIAL, not temporal

Date: 2026-08-31 · Author: MAIN · Cost: **$0** (three landed memos read at source + exact integer arithmetic)
Axis: no new measurement. `score_claim=false` · `promotable=false`
`verdict_scope`: **FORMULATION** — per-PAIR routing between the HG1 generator and the RC64 token
stream, on lb1's field. Does **not** close spatial routing; §4 names that cell and hands it on.

## 1. Why this cell was worth entering

`tba1` (#1234) closed subset-naming with a hard law: *"naming any subset costs more than the subset
holds."* But read its own denominator — `ddm_tba1:15,39`: **Gini 0.9951**, top 1% of POSITIONS holds
96.32%, and the cost/task mass occupy *"the same 0.2% of positions."* That closure lives at
**position granularity over 117,964,800 positions.**

At **PAIR** granularity the address is 600 bits = **75 B**, or **0.18% of the 42,097 B demand.**
The address tax is not the binding constraint here — which is exactly [[m124]]'s warning against
carrying one denominator into another, and rt3's #1342 correction that the address tax was POST-HOC
in 4 of 5 closures. So the cell is open on its own terms and had to be priced, not inherited.

## 2. THE EXACT ANATOMY — zero remainder (and a transcription error caught before publishing)

From `ddm_dcf1_duplicate_carry_factorization_20260831.md:49-56`, MEASURED, every span with a receiver
consumer:

| region | bytes |
|---|---:|
| ZIP framing | 100 |
| RX1 header | 14 |
| HPAC stream | 13,515 |
| semantic renderer | 30,856 |
| frame-0 carrier | 22,010 |
| residual correction table | 96 |
| **RC64 token stream** | **113,492** |
| **total** | **180,083** ✓ |

⚠ My first pass omitted the 14 B RX1 header and produced a 14 B hole — the exact
raw-vs-coded/omission class `gf1` §8a caught on itself today. Corrected at source before use.

Two exact consequences nobody had written down:

```
fixed (everything but tokens)   180,083 − 113,492 =  66,591 B   ← equals ar1b's residue EXACTLY
token budget at sub-0.12        137,986 −  66,591 =  71,395 B
token demand                    113,492 −  71,395 =  42,097 B   ← the WHOLE campaign demand
```

> **The sub-0.12 demand is entirely a TOKEN-STREAM demand.** The other 66,591 B are already inside
> budget. That is a cleaner statement of the target than "shed 42,097 B from the archive," and it
> falls straight out of dcf1's census once the anatomy is summed.

## 3. THE ROUTING ARITHMETIC — and it needs 64% of pairs

Route a fraction `f` of pairs to HG1's generator (47,603 B fitted to lb1's field, `gf1`) and the rest
to the RC64 token stream. HPAC must still ship in full while any pair uses tokens, so it stays fixed:

```
bytes(f) = 66,591 + 75 + 113,492·(1−f) + 47,603·f  =  180,158 − 65,889·f
bytes(f) ≤ 137,986   ⇒   f ≥ 42,172 / 65,889  =  0.6401
```

**64.01% of all 600 pairs must run on the generator.** Now the distortion, and this is where it dies:

| | measured |
|---|---:|
| lb1 token errors (DALI lineage, 0.00146%) | **~1,722** of 117,964,800 |
| HG1 mismatches on lb1's field | **1,325,033** (1.12324%) |
| token→argmax amplification | **1.157×, no attenuation** (`bz2d`) |

**Routing pays only if the error is CONCENTRATED — and `gf1` measured that it is not.** `gf1`
§2 verbatim: *"893,436 mismatches over 600 frames against a ~512-px boundary is **~1,489 px/frame** —
a horizon wrong by a few pixels."* The dominant 67.43% of the failure is a **per-frame systematic
shape error present in every frame**, not a heavy tail over pairs.

So the best 64% of pairs carry ≈64% of the mismatches ≈ **848,000** — a **~492×** increase in token
error over lb1, amplified 1.157× to argmax ⇒ d_seg ≈ **0.115**, S_seg ≈ **11.5** against a whole gap
of 0.028030. And the escape route closes too: paying `gf1`'s own measured **0.2909 B/correction** to
repair those 848,000 mismatches costs **246,683 B** — **3.46× the entire 71,395 B token budget**, on
top of the packet.

**Dead in both modes, by ~410× and ~3.5× respectively.**

## 4. THE LAW, AND THE CELL THAT SURVIVES

> **Routing needs a tail. `tba1` closed subset-naming because the address was expensive; per-pair
> routing dies for the opposite reason — the address is nearly free (75 B, 0.18% of demand) and there
> is nothing worth addressing.** Two closures, two mechanisms, and citing either one for the other's
> reason would be wrong.

This pre-closes every **temporal** routing granularity: per-pair, per-frame, per-block-of-pairs. A
uniform ~1,489 px/frame error has no temporal structure to exploit at any grouping.

**It does NOT close SPATIAL routing, and that cell is live.** `gf1` §8c prices the generator's four
streams and the asymmetry is extreme:

| stream | coded B | % of packet | % of gap |
|---|---:|---:|---:|
| lane | 36,044 | 75.7% | 24.03% |
| movable | 6,624 | 13.9% | 0.08% |
| **horizon** | **4,536** | **9.5%** | **67.43%** |
| mycar | 95 | 0.2% | 8.47% |

**The horizon stream buys 67.43% of the failure for 9.5% of the packet — it is cheap and wrong.** The
un-priced move is to keep the generator everywhere it is cheap AND right, and spend real tokens ONLY
on the horizon band. That is spatial, not temporal; its address is a band geometry the decoder can
compute, not a per-position list; and it is the "meet it where it lives" discipline applied to a
decomposition that already exists. **`gf1` §2 also measured the horizon error is SHAPE not SHIFT**
(per-frame |MEAN| 1.249 px vs per-column STD 3.523 px, ratio 2.819), so a scalar-per-frame fix
recovers only 6.1% — the band must carry real information.

Chartered as **`ddm_hzb1`** (see §6). Not measured here; named with its arithmetic and handed on.

## 5. Denominator

Cells entered: **1** (per-pair generator/token routing). Closed with mechanism: **1**. Granularities
pre-closed by the same law: **3** (per-pair · per-frame · per-block). Cells named-and-handed-on:
**1** (spatial horizon-band routing). New exact quantities derived: **3** (fixed = 66,591 B · token
budget = 71,395 B · required routing fraction f ≥ 0.6401). Own errors caught before publishing:
**1** (the omitted 14 B RX1 header). Measurements run: **0**. Dollars: **0**.

## 6. Fire order

`ddm_hzb1_horizon_band_spatial_route` — price the horizon-band spatial route: what does it cost to
carry ONLY the horizon band in real tokens on top of HG1's other three streams, and does the total
land under the 71,395 B token budget at lb1-class accuracy? Falsifier: the band's token cost plus the
remaining streams' residual exceeds 71,395 B ⇒ spatial routing closes too, and with it the whole
routing family at every granularity.

The exact pointer did not move. This unit did **not** achieve the goal — it entered an un-priced cell,
closed it with a mechanism that also pre-closes two sibling granularities, corrected its own arithmetic
before publishing, and left one measured, named successor.

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
