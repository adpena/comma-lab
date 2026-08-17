# ra2crr — the carrier drop question, PRICED over every direction: a bound, not a sample

`date_utc: 2026-08-16` · `owner: ddm_ra2_carrier_rank_refit` · `axis: [authority-tracking GT,
MEASURED 1.00081x vs contest-CUDA]` · `score_claim: false` · `promotable: false` ·
`frontier_moved: false`

Receipts: `/Volumes/APDataStore/pact/ddm_ra2crr/RA2CRR_PRICED_POSE_NULL.json` ·
payloads under `/Volumes/APDataStore/pact/ddm_ra2crr/retained/` (4 files, each with sha256 +
byte count in the receipt). Tool: `experiments/ddm_ra2crr_priced_pose_null_direction.py`.
Consumed banked artifacts only — **no render, no scorer forward, no dispatch, no launch.**

---

## THE ANSWER, FIRST

**My charter's premise was stale in three separate ways, and the family it assigned me was
already closed. I did not re-derive the closure; I converted its weakest link from a sample into
a bound, and corrected one load-bearing closure ground that does not survive at family level.**

1. **"Rung 2 has never been measured" — false.** `ra1` (10:29 today) measured it and closed it at
   FAMILY scope; `ra2c`, `jc1`, `ra2` and `ra3` each measured a distinct treatment the same day.
   Five treatments now span the space.
2. **"22,032 B basis+coeff pool" — a PR135-lineage figure that does not re-derive on hv1.** The
   basis half transfers (12,277 B); the coefficient half does not. On hv1 the coefficient stream
   measures 79,020 bits = **9,878 B**, not FD135's 78,036 bits = 9,755 B. **The live pool is
   22,155 B, not 22,032 B** — stale by 123 B.
3. **MAIN's own scope correction is superseded by the two memos it cites** (§6). Rung 2′ was
   measured and closed by `ra3` at 20:50, 4h23m after the recall that called it open; and the
   −15,157 B bar is the figure `fb1` exists to kill — the live bar is **−14,413.4 B**.
4. **NEW, and the point of this unit: the cheapest droppable direction ANYWHERE on the sphere
   costs `Δd_pose = 3.2824e-03`** against a break-even of `1.05e-06`–`2.19e-06`. That is a miss of
   **1,498×–3,139×** at face value, and **828×** even after granting the most favourable
   model-error factor ever observed on this object. 292 of 292 independent descents converge
   within 1% of that optimum, so this is a statement about the whole sphere, not a lucky sample.
5. **A correction that matters: the family is NOT closed by its ceiling.** `ra3` closed on
   "perfect execution returns 6.3–12.8% of the gap" — true for the **one-dimension** rung only.
   `ra1` MEASURED that rank-4 returns **14,709 B = 102.1%** of the gap. The family ceiling clears
   the bar. **The closure rests entirely on distortion**, and should be cited that way.
6. **MAIN's folded prerequisite (the carrier pose-sensitivity map) needed no new work — and its
   branch (b) fires.** Every one of the 12 carrier coordinates costs **24,835×–84,984×** break-even
   to drop; the best rotation reaches 1,498×. **No pose-null or pose-quiet carrier coordinate
   exists** (§5b).
7. **But the map does not unblock the banked rate candidates, because they are in a different
   section.** mp2's two refused candidates edit the **semantic renderer** (`blocks.{1,2,3}.film.weight`,
   "38/38 semantic tensors"), not the 22,161 B carrier — mp2 touched the carrier only via a
   *lossless* Brotli race that moved pose by zero. There is no anomaly to explain: `d_pose` reads
   both frames and the semantic renderer paints both. The live FD field is **semantic/FiLM**, as
   mp2 itself says twice; it needs a scorer pass, so I report it rather than fire it (§5b).

**Pointer UNMOVED: hv1 ep0634, S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600].**
This unit produced no lower score and did not attempt one.

---

## 1. Charter item 1 — the pool census, MEASURED, and the 22,032 B figure is stale

The archive parses section-for-section (two independent parses agree — `ddm_pz5…:45-56` parsed
the RX1 header of the frontier archive itself, `ddm_ra2…:70-82` agrees):

| section | bytes | share |
|---|---:|---:|
| ZIP overhead | 100 | 0.05% |
| RX1 header | 14 | 0.01% |
| HPAC model | 13,515 | 7.4% |
| semantic model | 34,763 | 19.0% |
| **CPR1/CAP1 carrier** | **22,161** | **12.1%** |
| residual table | 96 | 0.05% |
| token stream | 112,110 | 61.3% |
| **total** | **182,759** | **100%** |

**Five distinct "carrier" byte objects circulate. They are different objects and all are correct**
(`ra1:183`, `mp2:81`, `ra2:92`); confusing them is how the 22,032 B figure survived:

| object | bytes | source |
|---|---:|---|
| `carrier.br` — the shipped Brotli section (what every credit divides) | **22,161** | `ra1:183`; `ra2:78`; `pz5:54` |
| `carrier.raw.bin` — uncompressed compacted body | 22,219 | `ra1:183`; `mp2:81` |
| `outer_carrier.bin` / F0C1 with sparse selector | 22,242 | `JC1_PRODUCER.json` `archive_carrier_blob_bytes` |
| canonical CPR1 (PR130 shape restored) | 22,307 | `JC1_PRODUCER.json` `canonical_cpr1_bytes`, sha `709ea928c2d73c59…` |
| Brotli-q11 of canonical CPR1 = pricing baseline | 22,278 | `ra1:94`; `ra2:217` |

**The basis+coeff pool my charter names.** It is the logical payload *inside* the body:

| half | PR135 census (FD135) | hv1 MEASURED | delta |
|---|---:|---:|---:|
| basis codes (27,648 symbols, Huffman) | 12,277 B | 12,277 B | 0 |
| coefficients (7,200 values) | 9,755 B (78,036 bits) | **9,878 B** (79,020 bits) | **+123 B** |
| **pool** | **22,032 B** | **22,155 B** | **+123 B** |

The 22,032 B originates at `ddm_fd135…:126` as a census of the **PR135/F26** carrier, is relayed by
`rfo2:75`, named as a pool by `mp2:81`, and enters my charter via `ra2:93`. Two independent hv1
measurements put the coefficient half at 79,020 bits — `ra2` §5 (7,200 val × 10.9750 b/val) and
`ra3` §3 (post-overlay re-encode). **Verdict: 22,032 B is a cross-vehicle constant transfer.** Small
(0.55%), but it is exactly the genus the corpus keeps paying for, and it was asserted as
"different objects, all correct" without the coefficient half being re-derived.

**Which side of the placement law is this pool on? Both — and it cannot cut one without the other.**
The basis (12,277 B, 55.4%) is a *shared* dictionary the receiver applies to all 600 pairs —
model-side/in-network. The coefficients (9,878 B, 44.6%) are per-pair stored values — table-side.
Dropping one carrier dimension removes one basis atom **and** its 600 coefficients, and `ra3`
MEASURED the split of the resulting credit: **307 B from the basis, 635 B from the coefficients**
(the 11-atom rotated basis measures 11,970 B against 12,277 B shipped; the coefficient half returns
635 B). So **two thirds of every byte this family harvests is table-side, but it must spend
model-side capacity to get it.** hm1's law (`ddm_hm1_model_byte_derivative_20260816.md:150`) is that
in-network bytes are the productive ones — which is a structural reason this family's exchange rate
is poor. **I use the law's DIRECTION only and explicitly refuse to transfer its 26× magnitude:** hm1
measured 26× on the model↔token axis, not the carrier↔pose axis, and transferring that constant
across regimes is the bug the corpus has paid for repeatedly.

---

## 2. Charter item 2 — what was already measured, so I did not re-run it

Five treatments, all n600, all landed today. I re-derived none of them:

| treatment | owner | result | scope |
|---|---|---|---|
| α=0 (carrier deleted) | ra2c | 350,428× pose | FORMULATION |
| rank-r Frobenius truncation, r=1…11 | ra2c §8.1 / ra1 | misses 32.2×–145.3×, monotone, flat spectrum, no knee | FORMULATION |
| coordinate keep-set + coefficient re-fit | jc1 | 235.3× / 238.9× | FORMULATION |
| pose-metric subspace projection, r=11 | ra2 | 111.2× (cost ÷ credit) | INSTANCE |
| **subspace + trust-regioned per-pair re-fit, realised-accepted** | **ra3** | **35.5× — the best any carrier arm produced** | family closed |
| common pose-null `K = dim(∩ᵢ null(Jᵢ))` | jc1 | **K = 0** at every tolerance ≤ 3% | — |

`ra3`'s cell is rung 2′ exactly: rank reduction **plus coefficient re-fit**, per-pair, accepted on
realised measurement. It beat every predecessor by 3.13× and still refused by 35.5×.

---

## 3. What was actually left open — and the measurement that closes it

`K = 0` is a **rank test on the stacked Jacobian**. It answers "is any direction *exactly*
pose-free" and nothing else. Two quantities that decide the score are invisible to it:

- **the per-pair coefficient weighting.** Dropping `v` perturbs pair `i` by `δᵢ = −(zᵢ·v)v`; damage
  is weighted by how much each pair actually *uses* `v`. `σ_min` of the stacked Jacobian cannot see
  this. jc1's RMS-whitened coordinate is a diagonal approximation of the weighting, not the
  weighting.
- **the residual cross-term.** `d_pose` is measured against a GT the base render already misses, so
  damage is `2⟨rᵢ,pᵢ⟩ + |pᵢ|²`. The cross-term is **first order and can be negative** — a drop can
  move pose *toward* GT. `ra3` §7b measured that ignoring it misprices by **2.64× precisely near
  break-even**.

So the priced question — *what is the cheapest direction to drop, and what does it cost?* — was
open. It is answerable at $0 from banked artifacts, which `ra2c` §8.4 did not expect
("Not $0, not paid"); jc1's Jacobian had landed by then.

### The exact objective

For unit `v ∈ R¹²`, with `aᵢ = zᵢ·v`, `wᵢ = Jᵢv`, `rᵢ` the base residual against the authority GT:

    Δd_pose(v) = (1/(6N)) Σᵢ [ −2 aᵢ ⟨rᵢ, wᵢ⟩ + aᵢ² |wᵢ|² ]

Quartic on the sphere; minimised by Riemannian descent (renormalising each step — an unconstrained
`v = u/‖u‖` parametrisation has an exactly flat radial direction and L-BFGS drifts the norm to
0/inf, which is a defect I hit and fixed rather than worked around).

### Controls

| control | result |
|---|---|
| authority base `d_pose` reproduces ra3's `6.88559506e-06` | **2.60e-10** relative |
| every banked array all-finite | pass |
| per-pair Gram rank (`rank(JᵢᵀJᵢ)`) | **exactly 6 for 600/600 pairs** |
| analytic gradient vs central finite difference at the optimum | **4.95e-11** relative |
| optimum ≤ best structured direction (refuses otherwise) | pass |
| 292 descents from 12 axes + 12 singular vectors + 12 Gram eigvecs + 256 random starts | **292/292 within 1% of the optimum** |

The last row is the load-bearing one: the landscape is effectively unimodal, so `3.2824e-03` is a
property of the sphere, **not** the best of a lucky sample.

### The result

| direction | `Δd_pose` |
|---|---:|
| best spectral direction (`σ_min` right-singular vector ≡ smallest mean-Gram eigvec — the two agree, cross-validating jc1's `cond(J) = 12.02` against `ra2`'s `cond(G_pose) = 144.437 = 12.018²`) | 6.3213e-03 |
| **true minimum over the sphere** | **3.2824e-03** |
| worst direction | 2.9186e+00 |

**The cheapest direction is not any spectral direction — it is 1.93× cheaper than `σ_min`'s.** That
is the coefficient weighting and the cross-term doing exactly what §3 predicted they would. Anyone
ranking carrier directions by the Jacobian spectrum is ranking by the wrong quantity.

---

## 4. Pricing — the ceiling FIRST, per the charter

Gap to target: `S − 0.15 = 0.009597292954985986`, i.e. **14,413.4 B** at
`25/37,545,489 = 6.658589531221714e-7` S/B.

**Ceiling of the one-dimension rung** (what my measurement is about) — if the direction were
*exactly free*:

| credit convention | bytes | credit S | % of the gap | break-even `Δd_pose` | miss at my optimum |
|---|---:|---:|---:|---:|---:|
| **measured container** (ra3, real CPR1 blobs) | 913 | 6.0793e-04 | **6.33%** | 1.0459e-06 | **3,138.5×** |
| most favourable (basis pro-rated) | 1,658 | 1.1041e-03 | **11.50%** | 1.9542e-06 | **1,679.7×** |
| ra2 assumed uniform (kept for comparability) | 1,846.75 | 1.2297e-03 | **12.81%** | 2.1920e-06 | **1,497.5×** |

**Ceiling of the FAMILY** — and this is where I correct `ra3`. Its closure ground 2 reads "perfect
execution returns 913–1,847 B = 6.3–12.8% of the gap", which is the r=11 rung. `ra1` MEASURED the
whole ladder through the shipped coder:

| rank | bytes returned | % of the 14,413.4 B bar |
|---:|---:|---:|
| 11 | 1,667 | 11.6% |
| 8 | 7,587 | 52.6% |
| 6 | 11,049 | 76.7% |
| **4** | **14,709** | **102.1%** |
| 1 | 20,338 | 141.1% |

**The family ceiling clears the bar.** `ra3`'s ceiling argument is correct at r=11 and does not
generalise; the family is closed by **distortion alone** (rank-4 measured `d_pose = 0.354`,
2,400× base). This matters because a ceiling argument closes a family "for free" and this one does
not hold at family level — cite distortion, not the ceiling.

### Is my model number a safe bound?

**Not by assumption — by measurement.** I calibrated the linear model against all 12 realised ra3
candidates on the authority GT (no new forwards; pure arithmetic on retained `pose6`):

| candidate | step / coeff RMS | predicted `Δd_pose` | realised `Δd_pose` | realised ÷ predicted |
|---|---:|---:|---:|---:|
| projection | 0.1350 | 3.7585e-03 | 2.0865e-03 | **0.555** |
| mu100 | 0.1351 | 3.0483e-03 | 1.6866e-03 | 0.553 |
| mu30 | 0.1354 | 2.0146e-03 | 1.1800e-03 | 0.586 |
| mu10 | 0.1361 | 8.8115e-04 | 8.2078e-04 | 0.932 |
| mu3 | 0.1373 | 2.5409e-04 | 9.5531e-04 | 3.76 |
| mu1 | 0.1380 | 1.4381e-04 | 1.2650e-03 | 8.80 |
| mu0.1 | 0.1384 | 1.4103e-04 | 1.4911e-03 | 10.57 |
| mu0.001 | 0.1491 | 1.3926e-04 | 4.0275e-03 | 28.92 |
| mu0.0001 | 0.2207 | 1.3713e-04 | 5.9255e-02 | **432.10** |

**The model error is two-sided, and it is not a function of step size.** Step size is nearly
constant (0.135–0.14) across a ratio range of 0.55× to 432×. What varies is **whether the model
designed the step**: loose radii (the model barely consulted) give 0.55×, and the model *overstates*
damage; tight radii (the model in full control) give up to 432×, and the model *understates*
catastrophically. This independently reproduces jc1's structural finding — "a linearization used as
a screen is decent; used as a designer it over-estimates its own control authority by three orders
of magnitude" — with 12 points instead of 4, on the authority axis instead of the advisory one.

My `v*` is model-designed, so it sits on the *understating* side by that mechanism. But I do not
need that: **granting the single most favourable factor ever observed (0.553×, i.e. assuming the
model overstates by the maximum measured amount), the optimum still costs 1.816e-03 — a miss of
828× at the most generous credit and 1,737× at the measured one.** The closure survives the
most favourable admissible reading of its own instrument.

---

## 5. Why the RE-FIT half can never be closed by a model — a structural result

Dropping `v` and re-fitting the surviving 11 coefficients means solving
`rᵢ + JᵢP w = aᵢ Jᵢ v` for `w` in `v`-perp. **MEASURED: `rank(JᵢP) = 6` for all 600 pairs at all 8
random directions probed** — 6 pose constraints against 11 free coefficients.

**So the first-order re-fit is exactly solvable for every pair and every direction: the
model-optimal re-fit damage is identically zero, for all `v`.** No model-based bound on the re-fit
family exists — not a weak bound, *none*. Any re-fit verdict must be REALISED.

This is the structural reason jc1's designer failed by 1,065× (jc1 caught the same degeneracy at
r ≥ 6 and called it "a dimension count, not a result") and the reason ra3 had to build realised
per-pair acceptance. It also settles the epistemology of the closure: **`ra3`'s realised measurement
was not one option among several — it was the only admissible instrument**, and my §3 bound applies
strictly to the no-re-fit drop.

---

## 5b. MAIN's folded prerequisite — the carrier pose-sensitivity map is ALREADY MEASURED, and branch (b) fires

MAIN folded in mp2's RELAY 6: *"for each carrier row/coordinate group, perturb and measure d_pose
response; output a pose-sensitivity map over the carrier"*, with two branches — pose-null groups
exist (re-select the banked rate candidates onto them) or they do not (close Stage 1 honestly with
the map as evidence).

**The map exists and required no new work.** jc1's `J ∈ R^{600×6×12}` *is* a per-coordinate pose
response measured through the real chain, and my §3 objective converts it into the quantity the
branch actually needs — the cost of *removing* a coordinate, coefficient-weighted, cross-term
included. Dropping each carrier coordinate entirely, on the authority GT:

| carrier coordinate | `Δd_pose` | × break-even (1,846.75 B) |
|---:|---:|---:|
| 0 | 5.4437e-02 | 24,835× |
| 1 | 1.0183e-01 | 46,457× |
| 2 | 1.2565e-01 | 57,323× |
| 3 | 1.2208e-01 | 55,695× |
| 4 | 7.1828e-02 | 32,769× |
| 5 | 9.6127e-02 | 43,854× |
| 6 | 8.5191e-02 | 38,865× |
| 7 | 1.4317e-01 | 65,314× |
| 8 | 1.7236e-01 | 78,633× |
| 9 | 8.3257e-02 | 37,983× |
| 10 | 1.5265e-01 | 69,643× |
| 11 | 1.8628e-01 | 84,984× |

**MEASURED verdict: branch (b). There is no pose-null or pose-quiet carrier coordinate.** The
quietest single coordinate is **24,835× over break-even**; the quietest *rotation* — the true
optimum over the whole sphere, §3 — reaches 1,498×, still three orders out. jc1's `K = 0` said no
coordinate is *exactly* free; this says none is *affordably* free, which is the question that
decides work. Stage 1 closes on the carrier with a measured map as the evidence, exactly as MAIN
asked, and not one further n600 run on this family is warranted.

### The region misattribution — this map does NOT unblock the banked rate candidates

MAIN's crux was: *"If the pose carrier is a separate frozen section, why does re-precisioning
q3/q4 move d_pose by ~5×? Something the pose decode reads is inside the region those candidates
touch."*

**MEASURED at source: the two refused candidates do not touch the carrier.** From mp2's own memo —
FiLM keep87 prunes rows of `blocks.1.film.weight`, `blocks.2.film.weight`, `blocks.3.film.weight`
(`mp2:55`), and every generation is validated by "38/38 independently decoded **semantic** tensors"
(`mp2:38`). Both candidates are **semantic-renderer** edits (34,763 B section), not CPR1 carrier
edits (22,161 B section). mp2 touched the carrier exactly once — a *lossless* Brotli q0–q11 race
that decoded byte-identically 12/12 and moved pose by exactly zero (`mp2:81-89`).

**So there is no anomaly to explain.** `d_pose` reads *both* rendered frames; the semantic renderer
paints both, so re-precisioning it moves pose as a matter of course. The carrier renders frame_0
only — which is why it is seg-invisible (measured 3×) but *not* pose-invisible. Two sections both
feed pose; only one of them was touched by the refused candidates.

mp2 says this itself and MAIN's relay inverted it: *"The open field is **semantic/FiLM** sensitivity
through the render"* (`mp2:116`) and *"Re-running pz4a on the pose coefficients is closed … MP2
needs the semantic/FiLM-through-render field instead"* (`mp2:155`). There are **two distinct FD
fields**; mp2's carrier-rank fire trigger (`mp2:99`) names the carrier one, and its live blocker
names the semantic one.

**Consequence, and it is the actionable part:** the banked mz2 candidates (−823 B mixed q3/q4;
−130…−2,051 B FiLM rows; up to −2,874 B ≈ 20% of the gap) are blocked by pose collateral **in the
semantic/FiLM region**. A carrier map cannot unblock them, and neither could any further carrier
work. Building the carrier map further would have been the exact "spent the work to reproduce a
known refusal" failure MAIN warned against — one region over.

### What the semantic/FiLM map needs — reported, NOT fired

Per MAIN's discipline (*"if the probe genuinely requires a scorer pass, STOP and report"*): **it
does.** No banked Jacobian exists for the semantic/FiLM coordinates — jc1's is carrier-only (6×12
per pair), and there is no retained `pose6` stack for perturbed FiLM rows. An FD map over the three
FiLM tensors' rows requires, per perturbed row group, a full render of both frames plus a PoseNet
forward over the pair set. That is a scorer job and I did not fire it. **Cheapest honest form:** a
grouped FD (perturb row *blocks*, not single rows) with a screening pair subset, then n600 only on
surviving groups — but note the standing prefix-bias law, which is anti-conservative on the pose
axis specifically (pose prefixes measure 2.54–4.21× *harder* than the population), so any subset
screen must be a seeded RANDOM sample, never a contiguous prefix.

**On MAIN's base-figure caution — complied with, and the two numbers are genuinely different.** I
did not reuse any inherited pose base: I re-derived it from hv1's own retained `pose6_generated`
(produced from archive sha `80d9c8c6…` = hv1 ep0634 @ 182,759 B) against the authority GT cache,
obtaining **6.885595058208011e-06**, which reproduces ra3's hv1 figure to 2.60e-10. MAIN's flagged
CP135-pinned value is `6.885642960696714e-06` — different in the 6th significant figure
(7.0e-6 relative). The two are close enough to be mistaken for each other and are not the same
number; mine is hv1-derived and MEASURED.

---

## 6. Two stale premises in my own scope correction — reported, not worked around

MAIN's mid-arm correction directed me to two memos. Read at source, both contradict the direction
they were cited for. Chronology by commit date:

| artifact | landed | says |
|---|---|---|
| `ra1` | 10:29 | rung 2 closed at **FAMILY** scope, "re-fit included"; live bar **14,414 B** |
| `ddm_gestalt_two_week_recall` | **16:27** | rung 2′ (rank + re-fit) **"OPEN — the named successor"** |
| `ddm_fb1_stale_bar_rebase` | **19:12** | the **−15,157 B bar is STALE** (e480b base); live bar **14,413.4 B**; names `ra1`'s 14,414 B as its **positive control** |
| `ra3` | **20:50** | rung 2′ **MEASURED and closed** — 35.5× from break-even |

1. **"Rung 2′ is OPEN"** was true when written at 16:27 and was superseded 4h23m later by `ra3`.
   The recall's own honest limit says the truncation closure "does not bound the re-fit mechanism" —
   correct, and `ra3` then supplied the missing measurement. §5 above explains why it had to be a
   realised one.
2. **"The rate rung is −15,157 B, not any earlier figure"** inverts `fb1`, the memo cited alongside
   it. `fb1` exists precisely to kill that number: it is computed off the superseded e480b v2
   archive (183,502 B). Off the live 182,759 B the cut is **14,413.4 B** — `182,759 − 168,345.6`.
   The rung moves by exactly the pointer's byte move (743 B). This is the same genus as MAIN's own
   item 4 hazard (the stale 186,269 B trigger), one level up.

I flag both rather than silently using the corrected numbers, because a scope correction that
carries a stale constant will be re-issued to the next arm.

---

## 7. Honest limits

- **Every number here is a MODEL quantity on an STE-relaxed Jacobian.** jc1's `J` is
  `MEASURED_ON_STE_RELAXED_CHAIN` (forward exact and byte-identical 600/600; backward
  straight-through, since the exactly-quantized chain's Jacobian is 0 a.e.). Its finite-difference
  control covers **3 of 600 pairs** at ~3% agreement. My §3 optimum inherits that entirely.
- **§3 bounds the no-re-fit drop only.** §5 proves no model bound exists for the re-fit half; that
  half is closed by `ra3`'s realised measurement, which I did not re-run.
- **The calibration transfer is an argument, not a theorem.** I apply the most favourable *observed*
  ratio (0.553×) to a direction the model designed, and the mechanism I measure says model-designed
  steps sit on the *other* side. The 828× is therefore conservative by construction, but it is a
  transfer across candidates, not a per-candidate guarantee.
- **The 12-atom render is a shared fidelity limit** (inherited, not introduced): the receiver
  normalises atoms before the einsum, so a real 11-atom container would renormalise its own rotated
  atoms. Every exact `d_pose` in this family — ra2c's, jc1's, ra2's, ra3's, and my model — sits on
  the 12-atom render.
- **`ra3` is internally inconsistent by 5 B on its own headline credit** (§3 table says 918 B;
  abstract/§5/NEXT say 913 B; the 913 propagated into `ra2`'s banner). I priced at 913 B, the
  conservative end, so this does not move my verdict.
- **The rate credit is bracketed, not a point** — the shipped per-atom quantizer rule (three atoms
  cap at `|code| = 7`) is not reconstructible from the archive, so the rotated-basis size is an
  upper bound.
- **No contest row.** Advisory/model throughout; the authority-tracking GT is a retained local
  reference measured at 1.00081× the contest axis, not a contest measurement.

---

## 8. VERDICT

**`REFUSED — the carrier drop family is closed on the distortion axis, and the closure is now a
bound over all directions rather than a sample of constructions.`**

**`verdict_scope: FAMILY`** for the **no-re-fit rank/atom drop** of the shipped 12-dim carrier, in
**any** basis and under **any** direction: minimised over the entire sphere in the score-relevant
functional (coefficient-weighted, cross-term included, authority GT), the best case misses
break-even by 828× under the most favourable admissible calibration and by 1,498–3,139× at face
value. This strictly extends `ra2c` §8.1 (Frobenius-optimal, per-rank) and `ra2`'s INSTANCE scope.

**`verdict_scope: INSTANCE → deferred to ra3`** for the **re-fit** half: §5 proves no model can
bound it, so my instrument has nothing to say; `ra3`'s realised 35.5× is the operative closure.

**`Stage 1 (carrier pose-null admissibility): CLOSED, branch (b), MEASURED.`** No pose-null or
pose-quiet carrier coordinate or rotation exists (§5b). The banked rate candidates cannot be
re-selected onto the carrier, and their real blocker is the **semantic/FiLM** field, which is a
different section and needs a scorer pass this arm was correctly forbidden from firing.

**What would reopen it.** Not a new radius, a new basis, or a new rank — those are all inside the
bound. Only a change to the *object*: a carrier **retrained from scratch with pose in the training
loop** (rate-aware QAT), which `ra1` explicitly left outside its FAMILY verdict and which is
untested by construction. My bound does not touch it, because it is a statement about dropping
directions from *this* trained carrier.

---

## 9. NEXT_IF_RESUMED

| # | row | owner | fire-condition |
|---|---|---|---|
| 1 | **Nothing further on the carrier drop axis.** Six treatments, a per-coordinate pose map, and a sphere-wide bound now span it. | — | do not reopen without a retrained-with-pose carrier |
| 1b | **The semantic/FiLM FD pose map is the live blocker for −2,874 B (20% of the gap)** — mp2's banked q3/q4 and FiLM-row candidates are held on pose collateral in the SEMANTIC section, not the carrier. Needs a scorer pass (render both frames + PoseNet per perturbed row group); cheapest honest form is grouped-FD on a seeded RANDOM pair subset (never a contiguous prefix — pose prefixes measure 2.54–4.21× harder), then n600 on survivors only. | unowned — MAIN to route | a scorer lane frees; NOT $0 |
| 2 | **Rebase the −15,157 B bar wherever it still routes work.** `fb1` tracked five sites; MAIN's live scope correction is a sixth, and it is the one that reaches new arms. Live bar: **14,413.4 B**, invariant form `archive ≤ 168,345.6 B`. | MAIN | $0, immediate |
| 3 | **Correct `ra3` §5 ground 2 at source.** Its ceiling ground is r=11-scoped; `ra1` measured the family ceiling at 102.1% of the gap. The closure is sound but should cite distortion, not ceiling. | MAIN | $0, immediate |
| 4 | **Re-measure the 123 B coefficient-half drift** if any future rung prices against the 22,032 B pool. hv1 measures 9,878 B twice independently. | unowned | before the pool is priced again |
| 5 | **Port the calibration method, not the constant.** "Predicted vs realised across every retained candidate" costs zero forwards when payloads are kept, and it converted a naked linear model into a quantified two-sided error law here. It is the right instrument for any rung with a linear surrogate. | unowned | any arm about to trust a linearization |
