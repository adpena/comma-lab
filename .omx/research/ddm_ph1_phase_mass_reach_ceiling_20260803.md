# ddm_ph1 — F7 TEMPORAL/PHASE: the reach ceiling, measured and priced

- **arm:** ddm_ph1 (first owner, seg family F7, task #932) · **date:** 2026-08-03
- **axis:** `[macOS-CPU advisory]` NON-PROMOTABLE. `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`. **Pointer UNMOVED.**
- **cost:** $0. No scorer forward pass fired (n600 scorer slot held by `ddm_r1c`/`ddm_bc1`;
  lane claimed as array-reduction-only so it does not contend).
- **n / selection:** **n600, `selection_mode=all`** for every headline number. Sub-runs are
  `stratified` seed 7, never a prefix (m88/m96: a prefix is a different population, and the
  sign of the bias inverts between seg and pose).
- **denominator for every ΔS:** `gap_decomposition_against_floor_20260802`, live-best `pu2`
  S = 0.7910689 → PR130 bar 0.172141, **total gap 0.6189279** (seg 0.401519 = 64.9%).
  W = **1.2731082153320312** B/flip, verified at source
  (`test_gap_decomposition_against_floor.py:367`, 4·37,545,489/117,964,800).
- **artifacts:** `tools/ph1_phase_mass_reach.py`, `tools/ph1_offset_coder_race.py`
  (committed `9e23e3fa41`, `4c2157f1dd`); receipts under
  `/Volumes/VertigoDataTier/pact/ddm_ph1_20260803/`.

---

## §0 Headline

**Whole-frame rigid phase is refuted; REGIONAL phase is a large paying lever.**

At n600 the best coder-closed rung is a **block8 dense offset field at 175,972 B (SMEVR)**
removing 270,755 of 458,738 flips: gross **ΔS_seg +0.22952**, rate cost **−0.11717**,
**NET +0.11235 S = 18.2% of the entire remaining gap.** block16 is within noise of it
(+0.11186 at 64,953 B) and is the better risk-adjusted rung because it costs 2.7× fewer bytes.

| rung (partition) | flips left | gross reach | best coder | bytes | ΔS_seg | ΔS_rate | **NET ΔS** |
|---|---:|---:|---|---:|---:|---:|---:|
| global (whole frame) | 458,716 | **0.005%** | — | — | +0.00002 | — | ~0 |
| component | 446,604 | 2.65% | — | — | +0.01029 | — | — |
| block64 | 406,515 | 11.38% | smevr | 5,967 | +0.04427 | 0.00397 | +0.04030 |
| block32 | 355,472 | 22.51% | smevr | 22,670 | +0.08754 | 0.01510 | +0.07244 |
| **block16** | 275,766 | 39.89% | smevr | **64,953** | +0.15511 | 0.04325 | **+0.11186** |
| **block8** | 187,983 | 59.02% | smevr | 175,972 | +0.22952 | 0.11717 | **+0.11235** |
| block4 | 110,126 | 75.99% | smevr | 489,364 | +0.29552 | 0.32585 | −0.03033 |
| block2 | 67,947 | 85.19% | lzma | 1,043,248 | +0.33128 | 0.69466 | −0.36338 |
| per-pixel oracle | 50,607 | 88.97% | — | — | +0.34598 | — | NOT SHIPPABLE |

**⚠ The single caveat that bounds all of it is in §6: this is measured by translating the
ARGMAX field. A real receiver translates TOKENS and the argmax emerges through
R→uint8→SegNet. §6 is the named next measurement and no byte of this may be banked before it.**

---

## §1 Apparatus validity — checked before anything was read off it

The residual argmax field is recoverable with **no scorer pass**: `R == G` everywhere except
the `ru1` atlas flip rows, where `R == realized_class`. Reconstruction from cached GT
`lstars` + `atlas_flat.npz` gives:

- **positive control: 458,738 reconstructed flips vs 458,738 atlas rows, absdiff = 0.**
- **mutation check (si1): perturbing one pixel moves the count — the control CAN go red.**
- **second, independent control:** the per-edge baseline this probe computes reproduces the
  `ru1` receipt's `top_class_pairs` exactly — Lane→Road **176,733**, Road→Lane **49,107** —
  without ever reading those fields.
- **subset validity:** n120 stratified seed 7 reproduces the n600 block8 reach to **0.14 pts**
  (59.16% vs 59.02%).

**The exactness trick.** For |dy|,|dx| ≤ rmax a pixel cannot change agreement if `R` is
constant on its (2·rmax+1) box **and** `R==G` there (then `R(y+dy,x+dx)=R(y,x)=G(y,x)`). So
restricting the sweep to the active band is **exact**, not an approximation; every count
reported is a full-field count.

---

## §2 A structural fact about the lattice that was not previously stated

**The GT argmax field is exactly constant outside rows 154–297, for all 600 pairs, with zero
variance:** rows 0–153 are entirely class 2 (Undrivable/sky), rows 298–383 entirely class 4
(MyCar/hood). Measured directly, not inferred.

- **Only 144 of 384 rows — 37.5% of every frame — carry any task content at all.**
- The `ru1` atlas's y-range [154, 297] is therefore **real geometry, not a filtered mine.**
  (I checked this before trusting the atlas; a filtered mine would have made every number
  below band-scoped and uninterpretable.)
- Flip density **within** the active band is 458,738/44,236,800 = **1.037%**.

This fact is the **mechanism** behind §3.

---

## §3 Whole-frame rigid translation is REFUTED — and now has a mechanism

**22 of 458,738 flips removable — 0.0048% reach.** `frac_cells_choosing_zero_shift = 99.7%`:
599 of 600 pairs prefer no shift at all.

This **independently replicates `gt2x`** (which measured 0.009% by a different route) using a
different apparatus, a different data path, and an exhaustive ±5 sweep. Two apparatuses, same
verdict.

**The mechanism §2 supplies:** the frame contains two large *static* slabs. A global shift of
dy translates the sky/hood seams by dy, manufacturing ~512·|dy| flips per seam — more than the
~765 flips/pair that exist in total. **The static slabs pin the global phase.** Any carrier
that applies one rigid transform to the whole frame is fighting 62.5% of the lattice that must
not move.

> **verdict_scope: FAMILY (whole-frame rigid translation), n600, mode=all, exhaustive ±5.**
> This kills frame-rigid *translation*, not positional DOF as such — §4 shows the opposite for
> regional forms. It does not test rotation, scale, or an expansion field, which `gt2x`
> separately argues is the true ego-motion form.

---

## §4 Regional phase pays — the coherence length is ~8–16 px

Reach is monotone in granularity and the **economics turn over between block8 and block4**.
The interesting quantity is not the reach but *where the reach stops being worth its bytes*:
below ~8 px the offset field costs more than the flips it buys, because the number of cells
grows as 1/K² while reach grows only sub-linearly.

**SMEVR wins every rung**, by 15–24% over the best generic coder (block16: 64,953 vs lzma
80,680 = −19.5%; block8: 175,972 vs 207,520 = −15.2%). This is the shipped r7 coder that
byte-closes the archive, so these are matched-currency bytes.

---

## §5 Three measured results on the operator's address-solve directive

**5a. The where-tax does not bind this carrier — measured, not asserted.** `gt2x` measured
~78% of explicit-production bytes are WHERE. A dense offset grid pays that at **zero by
construction**: position is implicit in raster order, so every coded byte is WHAT. I raced it
rather than assuming: dense-implicit vs sparse-explicit (cell_index, dy, dx) for non-zero
cells only.

| rung | dense-implicit (best) | sparse-explicit (best) | uncoded where-fraction of sparse |
|---|---:|---:|---:|
| block64 | **7,460** | 8,919 | 33.3% |
| block32 | **27,888** | 30,277 | 33.3% |
| block16 | **80,680** | 81,740 | 50.0% |
| block8 | **207,520** | 207,848 | 50.0% |

*(generic-coder columns, matched, so the comparison is like-for-like; SMEVR does not apply to
the heterogeneous sparse record stream.)* Dense wins at every paying rung. **For a dense phase
field the address-solve is already solved by the form** — which is the operator's
generative-addressing point, made empirically. Modes 1–6 of the cheap-address stack target a
cost this particular carrier does not pay; they remain live for the *sparse* carriers.

**5b. ξ-transport of the offsets LOST at every rung.** Coding pair p's field as an innovation
against pair p−1's was 15–18% *worse* (block16: 95,548 vs 80,680 lzma). Order-0 entropy went
*down* under differencing (999,723 → 974,497 bits) while real coder output went *up* — so the
redundancy the coders were exploiting is **spatial**, and temporal differencing destroys it.

> **verdict_scope: FORMULATION (offset-field differencing at pair-index adjacency, modular
> wrap, n600).** This is NOT a refutation of ξ-transport. Three unexamined confounds: (i) I did
> not verify that consecutive pair indices are temporally adjacent — if pair order is not
> temporal, temporal prediction cannot work *by construction*; (ii) modular wrap may defeat the
> coders' modelling; (iii) transporting the *generators* (operator's mode-6 multiplier) is a
> different operation from differencing the *offsets*, and was not tested. Any of the three
> could flip it.

**5c. My own silent-instrument bug, caught and fixed.** SMEVR reported `None` for the entire
first race — a bare `except: return None` swallowed a **sys.path** import failure, which read
as "SMEVR does not apply to this stream." It in fact **wins outright**. This is exactly the
m50 vacuity class, committed by me, inside the arm whose charter cites m50. Fixed: every
failure reason is now recorded in the receipt (`smevr_failures`), so a missing number can
never again be mistaken for a measured non-result.

---

## §6 THE BOUNDING CAVEAT — this is an argmax-field result, not yet a realized one

Every number above is measured by **translating the realized argmax field directly**. A real
receiver does not have an argmax field to translate: it has tokens, renders them through
**R → uint8 → SegNet**, and the argmax emerges at the far end. Translating tokens is *not* the
same operation as translating the argmax.

So the honest status is: **+0.11235 S is the reach ceiling of regional phase correction,
conditional on the shift being realizable through R.** It is an upper bound on the carrier's
delivered value and a lower bound on nothing.

Per the operator's realization doctrine — *"realization is always possible with the right
engineering"* — a disappointing realized effect here would be **MECHANISM-scoped, not a family
verdict**, and the response is the `sq1` method: stage-decompose (paint / R·D / uint8 measured
exact), isolate the debt to one stage, re-engineer that stage (recall that solve-from-frozen-
operator beat paint-truth, +0.7895 vs −3.764). That is the next measurement, not a caveat to
be worked around.

---

## §7 Instrument capacity (LAW A — a negative measures the instrument)

The sweep sees only |shift| ≤ rmax, so reach is **censored from below**. Measured directly with
a **matched** control (same n120 stratified seed-7 subset, only rmax varied):

| rung | rmax=5 | rmax=9 | censoring |
|---|---:|---:|---:|
| block16 | 40.19% | 44.41% | +4.22 pts |
| block8 | 59.16% | 63.99% | +4.83 pts |

**The rmax=5 reach numbers are lower bounds by ~4–5 percentage points.** The censoring is
modest, so the ceiling is **family-scoped, not instrument-dominated** — but it is not zero and
the headline should be read as conservative. On-rim fractions at rmax=5 are large (block8
61.4%, block4 73.8%), consistent with that.

**Coder-closed pricing is capped at rmax ≤ 7**, because SMEVR accepts at most 16 levels and the
alphabet is 2·rmax+1. Widening beyond that needs a different coder, so the priced numbers in
§0 are the coder-closed ones and cannot simply be extrapolated to rmax=9.

---

## §8 Candidate verdicts (charter (a)–(d)) — every row FIRED, FOLDED, or QUEUED-WITH-FIRE-ORDER

| # | candidate | status | disposition |
|---|---|---|---|
| (a) | ξ-keyed warp of the token field (#774/QA39 receiver expert) | **global form REFUTED** (§3); regional token-level form UNTESTED; ξ-transport of offsets LOST (§5b, formulation-scoped) | **QUEUED — fire order 2.** Fire only as a *regional* operator, and test generator-transport (mode 6) rather than offset-differencing. |
| (b) | per-pair boundary offset δ(s) along the separatrix (of1 P2C-OF) | **STRONGEST.** The block ladder is a coarse 2-D discretisation of exactly this and it pays at +0.11235. A 1-D arc-length parametrisation carries the same displacement with ~half the DOF. | **QUEUED — fire order 1** (below). |
| (c) | per-pair phase index into a shared codebook (LOTTO / p3v2) | **NOT RACED.** I did not locate LOTTO machinery by filename and did not have budget to recall it at source. Reporting this as un-run, not as a negative. | **QUEUED — fire order 3.** Must win at matched bytes vs the §0 SMEVR dense field, never adopted by reputation. |
| (d) | derived: dense regional offset field | **FIRED — this memo.** It is the thing that pays. | Landed. |

**Fire order 1 (the one I would run next, and why).** Re-run the ladder with the offset
restricted to the **local boundary normal** — 1 DOF per cell instead of 2. Rationale: if the
residual is boundary *displacement*, the tangential component is pure waste, and halving the
alphabet should cut bytes ~2× at near-equal reach, moving block8 from +0.112 toward +0.17.
This is the cheapest test that separates "phase = boundary displacement" from "phase = generic
2-D jitter", it reuses the committed sweep unchanged, and it is the discrete precursor to both
(b)'s δ(s) and the operator's diagram-native form (generator-pair ID + arc-length offset,
where the carrier and the address system are the same object).

**Explicitly NOT claimed:** no pointer movement; no realized-through-R number; no promotion.
The §0 table is a reach ceiling with a named, unmeasured realization step (§6).

---

## §9 What I refute in my own charter

1. **"the cheapest per-pair positional carrier"** presumes per-*pair* granularity. Measured:
   per-pair is worth **0.005%**. The unit is per-*region* at ~8–16 px. The charter's framing
   would have pointed the work at the one granularity that does not work.
2. **The charter's W-discrepancy note (#921: 1.2742 vs 1.2731)** — resolved at source. W is
   *derived*, not a constant: `4·rate_denominator/(600·512·384)`. 1.2731082153320312 follows
   from denominator 37,545,489. Any 1.2742 reading corresponds to a **different rate
   denominator** (~37,578,000), which is the Catalog #812 dynamic-`rglob` hazard — a stray
   `._*`/`.DS_Store` in `upstream/videos/` — not a competing calibration. Nothing to reconcile;
   the guard already landed.
3. **`ddm_pc2`'s "tokens_delta carries AMPLITUDE only" is confirmed and now sized** — the
   positional deficit is worth up to +0.115 S at coder-closed prices. But pc2's framing left
   the granularity open, and that was the whole question: the same deficit is worth ~0 at frame
   granularity and +0.112 at 8 px.
4. **rungC (per-pixel oracle, 88.97%) is near-tautological and I decline to headline it.** It
   asserts only that a correct-class pixel exists within 5 px, which is implied by 93.9% of
   flips lying on a GT boundary (ru1). Its offset alphabet (6.9 bits) is also *wider* than
   simply transmitting the class (2.32 bits), so it is dominated by direct class transmission
   at every pixel. It is the translation family's ceiling, never achievable headroom.
