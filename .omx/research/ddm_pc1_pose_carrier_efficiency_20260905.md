# ddm_pc1 — pose-carrier efficiency: basis precision, coefficient precision, generated basis, learned low rank, each WITH the re-solve

`[no-triality] [p0-ledger-ok]` · arm `ddm_pc1` · 2026-09-05 · lane `lane_ddm_pc1_pose_carrier_efficiency_20260905`

**Axes.** Bytes `[exact local byte arithmetic, receiver-verified parse-back]`. d_pose
`[macOS-CPU advisory, cpu_torch fp32 authority backend, n600, DALI GT]`. `score_claim=false`,
`promotable=false` for every row below: no contest-CUDA T4 row exists for any candidate here.

---

## 1. The object, re-derived from the bytes (MAIN's correction, honoured)

I did not carry any recalled carrier figure forward. I parsed `archive.zip` with the receiver's own
RX1 container reader (`runtime/residual_archive.py` in the frontier receiver copy) and read every
number out of the bytes. **Two accounting bases exist and they differ; every count below says which.**

**Archive basis** — what the rate term charges (`upstream/evaluate.py:63` stats `archive.zip`):

| RX1 section | bytes | share of 179,982 |
|---|---:|---:|
| ZIP overhead (one STORED member `p`) | 100 | 0.06% |
| RX1 model header | 14 | 0.01% |
| HPAC stream | 13,466 | 7.48% |
| semantic (SM3R renderer) stream | 30,856 | 17.14% |
| **carrier stream (the pose carrier)** | **22,031** | **12.24%** |
| section tail (token stream) | 113,515 | 63.07% |
| **total** | **179,982** | 100.00% |

The six rows sum to exactly 179,982 — MEASURED, not reconciled.

**Container basis** — the carrier stream is `brotli(q=9, lgwin=16)` over a 22,278 B body, after the
RR5 adaptive-arithmetic basis rider (RX1 `reserved = 0x1a`, bit `0x08` set):

| carrier body field | bytes | note |
|---|---:|---|
| u24 basis_bits + u24 residual_bits | 6 | |
| scales (12 basis f32 + 12 coefficient f32) | 96 | |
| packed CAP1 metadata (AR1 factors/biases, Huffman lengths, Rice k) | 40 | |
| **basis payload** | **12,277** | canonical Huffman, 27,648 five-bit zigzag symbols |
| **coefficient payload** | **9,830** | AR1-predicted, Rice-coded, 600×12 signed int12 |
| frame-0 selector tail | 29 | |
| **body total** | **22,278** | → 22,031 B stored |

**Brotli removes only 247 B (1.1%) of that body**, because both payloads are already entropy-coded.
That is the number that makes the whole arm tractable: **carrier payload savings pass through to
`archive.zip` at very nearly 1:1**, so the container basis and the archive basis agree on margins even
though they disagree on totals. MAIN's charter total (22,010 B) and my parse (22,031 B) differ by 21 B;
the parse is authoritative because it sums to the archive exactly.

**Basis geometry (MEASURED, from `decode_compact_carrier`).** 12 atoms × **3 planes** × 24 × 32 =
27,648 symbols; codes span `[-15, +15]`, 31 of the 32 alphabet symbols used. *The charter calls this a
"12-dim luma basis"; it is not luma* — per-plane spread from the plane mean is 0.4927, so the atoms
carry real chroma structure. Per-atom `|code|` max is `[15,15,7,15,15,7,15,15,15,7,15,15]`: **three of
the twelve atoms (2, 5, 9) already fit a 4-bit alphabet with zero loss.**

**`basis_scales` is 48 dead bytes (MEASURED).** `cpr1/inflate.py:245-246` multiplies the basis codes by
a per-atom scale, and `normalized_basis` then centres and RMS-normalises *per atom*, so a positive
per-atom scale cancels exactly. All 12 shipped scales are positive; re-rendering the basis from the raw
codes alone and comparing to the shipped normalised basis gives a max abs difference of **1.9073e-06**
(float32 epsilon on an RMS-1 field). Those 48 bytes reach nothing. See ITEM 1.

**SVD of the 600 realized carrier fields** (`einsum(coeff, basis_norm)/√12` at 3×384×512, via the
600×600 Gram). Energy share: 0.5734, 0.2959, 0.0414, 0.0259, 0.0247, 0.0106, 0.0091, 0.0060, 0.0049,
0.0035, 0.0029, 0.0019. Cumulative at rank 8 = **0.98686**, rank 9 = 0.99172 (effective rank 9 at 99%),
rank 12 = 1.0 (the field is exactly 12-dimensional, as it must be). The coefficient matrix itself has
singular values 12.73 … 2.54, cumulative energy 0.9141 at rank 8.

---

## 2. The base row, and a cross-axis control worth keeping

| | value |
|---|---|
| body | cl2 repack, `archive.zip` sha `08ec8533…`, 179,982 B |
| container identity control | rebuilding the shipped codes through `ddm_up3.build_archive` reproduces sha `08ec8533…` **bit for bit** |
| basis re-encode control | re-encoding the decoded basis symbols reproduces the shipped 12,277 B payload and its 98,213-bit count exactly |
| **d_pose n600, cpu_torch fp32, DALI GT** | **6.134076407345324e-06** |
| pose leg √(10·d_pose) | 0.007832034478566424 |
| per-pair median / max | 1.0593e-06 / 2.1493e-04 |
| contest-CUDA T4 d_pose for the same body | 6.14e-06 |

**Instrument control (run after the fact, because I had been relying on it implicitly).** The variant
instrument rebuilds `basis_raw` from the decoded codes and scales rather than reusing the shipped
tensor, so a variant that changes nothing must reproduce the shipped basis exactly or every delta below
would be contaminated. MEASURED: for both `v0_base` and `v3_lattice10`, `basis_raw` and `basis_norm` are
**bit-identical** to the shipped state (max abs difference 0.0 on both). For V3 the coefficient scales
are exactly 4× the shipped ones, the projected start codes stay inside signed int12 (max |code| 512),
and they reconstruct the shipped coefficients to within **2.0 shipped lattice steps** — exactly the
±0.5-coarse-step rounding a ×4 coarsening must produce, so the projection is right by construction.

**The local advisory instrument agrees with the T4 axis to 0.1% on d_pose** (6.134076e-06 vs 6.14e-06).
That is a control, not a licence: it is one quantity on one body, and it does not make any number here
a score. But it does mean the pose arithmetic below is not being done on a differently-scaled axis.

Exchange rate: 1 archive byte = 25/37,545,489 = 6.658589531221714e-07 S.

---

## 3. PRIOR-LAW PREDICTION vs MEASURED

The charter's prediction structure stands; its numbers were DERIVED and are now replaced by
measurements. **Rate is MEASURED by building the real archive through the identity-controlled
container and stat-ing it — never estimated.** Break-even d_pose is computed against the MEASURED base
(6.134076e-06), not against the charter's operating point.

| variant | rate saving (charter, DERIVED) | rate saving (MEASURED) | ΔS_rate (MEASURED) | break-even d_pose | binding cap | d_pose n600 (MEASURED) | ΔS net |
|---|---:|---:|---:|---:|---|---:|---:|
| V1 basis 5→4 bits | −2,440 B | **−392 B** | −2.6102e-04 | 6.550e-06 (+6.8%) | break-even | not solved (see §5) | — |
| V2 basis 5→3 bits | −4,880 B | **−3,204 B** | −2.13341e-03 | 9.931e-06 (+61.9%) | break-even | **≥ 2.8487e-05** | **REFUSED, §5b** |
| V3 coefficients, lattice ×4 coarser | −1,200 B | **−1,814 B** | −1.20787e-03 | 8.172e-06 (+33.2%) | break-even | *pending* | *pending* |
| V4 GENERATED basis (rank 12, zero stored bytes) | −12,200 B | **−12,043 B** | −8.01894e-03 | 2.5125e-05 | **ft1 ceiling 1.694e-05** | **≥ 6.13 per pair** | **REFUSED, §4** |
| V5 learned rank-8 SVD basis | −4,070 + −3,280 B | **−3,161 B** | −2.10478e-03 | 9.882e-06 | break-even | not solved | **dominated, §6** |
| V2+V3 composition | — | **−5,018 B** | −3.34128e-03 | 1.2484e-05 (+103.5%) | break-even | ≥ 2.8487e-05 | refused with V2 |
| V1+V3 composition | — | **−2,206 B** | −1.46888e-03 | 8.706e-06 | break-even | not solved | — |
| V4+V3 composition | — | **−13,857 B** | −9.22681e-03 | 2.8657e-05 | ft1 ceiling | — | refused with V4 |

**Compositions are EXACTLY additive here, MEASURED and not assumed** ([[m164]]: union ≠ sum in general,
so it gets checked). At the optimal-form quantiser: V1+V3 = −392 + −1,814 = −2,206 measured −2,206;
V2+V3 = −3,204 + −1,814 = −5,018 measured −5,018. At the naive quantiser the sum is off by 2 B on one
row (V2+V3 −8,927 summed vs −8,929 measured). V4+V3 = −12,043 + −1,814 = −13,857 measured −13,857.
The reason additivity holds is structural rather than lucky: V1/V2/V4 rewrite the basis payload and V3
rewrites the coefficient payload, and the two are disjoint byte ranges of the same carrier body whose
brotli cell removes only 1.1% of it — there is almost no shared redundancy for the two edits to
double-count.

**Where ft1's ceiling binds.** ft1's measured same-object pose ceiling is 1.694e-05. For V1, V2, V3 and
V5 the break-even d_pose is *below* that ceiling, so break-even is the binding constraint and the
ceiling never engages. For V4 (and V4+V3) the break-even is 2.51e-05, *above* the ceiling: **the
ceiling binds first and V4's admissible window is d_pose ≤ 1.694e-05, not 2.51e-05.** V4 misses both by
orders of magnitude, so the distinction is academic here — but it is the rule I applied.

---

## 4. V4 — the generated basis is REFUSED, on a lower bound, not an extrapolation

**Design at optimal form.** I did not pick one generated basis and call it the test. Three fully generic
constructions were built and scored for how much of the shipped realized carrier field their span
contains — the cheapest available upper bound on what a generated basis can reproduce, computed before
any solver time was spent:

| generated basis (12 atoms, 24×32, lowest total-degree DCT frequencies) | explained energy of the 600 realized fields | Gram condition |
|---|---:|---:|
| `luma` — lowest 12 frequencies, identical on all 3 planes | **0.02491** | 1.0012 |
| `planar` — lowest 4 frequencies × 3 planes | 0.01550 | 1.0000 |
| `opponent` — lowest 4 frequencies × 3 generic colour directions | 0.01550 | 1.0000 |

`luma` is the best of the three and it spans **2.49%** of the shipped field. That is a coverage bound,
**not** a d_pose verdict: the carrier only has to steer six PoseNet outputs per pair, and there is no
law saying it must do so along the field the incumbent happens to use. The re-solve is the object
change ([[m148]]), so the bound cannot close the question. It had to be measured.

**MEASURED, `luma`, full `jg5.refine_pair` re-solve (40 outer rounds, 400 GN iterations, ±2 polish),
warm-started from the least-squares projection of the shipped field, on 20 STRIDED pairs
(0, 30, … 570 — never a contiguous prefix, because a pose prefix of this video measures 2.54–4.21×
harder than the population, [[m96]]). Every one of the 20 stopped at `no_improving_step`: a physical
stop where the GN direction and the ±2 neighbourhood both refused, not an iteration budget.**

| pair | start | final | pair | start | final |
|---:|---:|---:|---:|---:|---:|
| 0 | 56.437 | **52.498** | 300 | 60.177 | 38.361 |
| 30 | 74.789 | 29.235 | 330 | 27.015 | 14.817 |
| 60 | 120.348 | **117.077** | 360 | 38.004 | 15.300 |
| 90 | 57.147 | 21.421 | 390 | 37.601 | 14.307 |
| 120 | 18.492 | 11.553 | 420 | 53.532 | 49.638 |
| 150 | 27.789 | 13.122 | 450 | 80.304 | 50.594 |
| 180 | 20.870 | **6.132** | 480 | 27.962 | 11.995 |
| 210 | 49.792 | 19.705 | 510 | 55.588 | 53.823 |
| 240 | 57.396 | 17.072 | 540 | 27.121 | 21.304 |
| 270 | 33.457 | 9.758 | 570 | 57.421 | 31.466 |

Final d_pose min / median / max = **6.132 / 20.505 / 117.077**. Sum over the 20 = 599.179.

**Why 20 pairs settle this without an n600 mean.** d_pose is the mean of 600 *non-negative* per-pair
values, so the n600 mean is bounded below by the measured pairs' sum divided by 600 — a bound that
holds for the whole population and that no unmeasured pair can reduce:

    d_pose_n600  ≥  599.179 / 600  =  0.998631

against a break-even of 2.5125e-05 and an ft1 ceiling of 1.694e-05. That is **39,748× past break-even
and 58,952× past the ceiling**, and 162,808× the base d_pose. This is a completed argument on the
population, not a subset verdict: the falsifier fires on a lower bound, and it fires with six orders of
magnitude to spare. Even the single best pair (6.132) exceeds `600 × ceiling = 1.016e-02` by 603×, so
any *one* of the twenty would have been enough.

**verdict_scope: FAMILY — generated separable-DCT bases at rank 12 on this vehicle's carrier.**
Three generic colour placements were tried (achromatic, per-plane, generic-opponent), each re-solved to
a physical stop from a least-squares warm start. The mechanism is not a solver failure and not a warm-
start failure: the solver ran out of improving steps on every pair while still six orders of magnitude
high. What is NOT closed: non-DCT generated families (Zernike, wavelet, steerable, Gabor), higher-rank
generated bases, and generated-plus-small-stored-correction hybrids. The measured wall is that a smooth
low-frequency 24×32 field cannot produce the PoseNet response the carrier needs.
Since the prize is the largest on the board (−12,043 B = −8.02e-03 S), the family deserves one more
attempt with a basis chosen to match the *Jacobian* rather than the field. See ITEM 2.

---

## 5. V1 — refused on arithmetic, and the reason is a real exchange curve

V1 (basis 5→4 bits) is not worth 600 pair-solves, and the reason is itself the finding.

**The quantiser step is free, and choosing it exposes a rate↔fidelity exchange.** `basis_scales` is
per atom and cancels in `normalized_basis` (§1), so the quantiser step is a free per-atom parameter. A
global `codes/2` — the obvious first pass — is one point on that curve, and it is the byte-greedy end:

| 4-bit quantiser | per-atom cosine to the shipped atom (after upsample, centre, RMS) | basis payload | archive Δ |
|---|---|---:|---:|
| global `round(codes/2)` (naive) | 0.899 – 0.983 | 8,196 B | **−4,071 B** |
| per-atom searched step (optimal form) | **0.991 – 1.000** | 11,927 B | **−392 B** |

| 3-bit quantiser | per-atom cosine | basis payload | archive Δ |
|---|---|---:|---:|
| global `round(codes/4)` (naive) | 0.715 – 0.949 | 5,563 B | −7,113 B |
| per-atom searched step (optimal form) | **0.970 – 0.992** | 9,087 B | **−3,204 B** |

The searched step lands near 0.94 at 4 bits — that is, *clip the tails, do not rescale*, because the
code distribution is concentrated (per-atom std 1.44–3.77 against a ±15 range). Clipping preserves
fidelity and preserves entropy; rescaling destroys both. **The bit-depth knob is therefore a crude
handle on a two-sided curve, not a rate lever with a fidelity side-effect.** Any future basis-precision
work should sweep the per-atom step directly against ΔS rather than pick a bit depth.

**Why V1 is refused.** At optimal form V1 buys −392 B = −2.6102e-04 S, so it needs d_pose held within
**+6.8%** of the base. The naive-quantiser run (before the optimal-form fix) measured what basis
perturbation does to this carrier even with the full re-solve: pair 0 landed at 9.892e-04 and pair 60 at
1.972e-02 after `no_improving_step`. Pair 60 alone contributes 1.972e-02/600 = 3.29e-05 to the n600
mean — **5.4× the entire base d_pose** from one pair. The optimal-form quantiser is far more faithful,
but a lever whose whole budget is 6.8% of d_pose, on a carrier where single pairs can move the mean by
540%, is not where 600 pair-solves belong while V2 (61.9% budget) and V3 (33.2% budget) are unmeasured.

**verdict_scope: INSTANCE — V1 as chartered (4-bit alphabet, ΔS-priced at the fidelity-optimal step).**
Not a claim about basis precision as a family; §5's exchange curve is the family-level result, and it
says the interesting question is the step, not the depth. The two per-pair numbers quoted are from the
naive-quantiser build and are cited as *evidence that basis perturbation is dangerous on this carrier*,
never as V1's d_pose. V1 has no measured n600 d_pose and none is claimed.

---

## 6. V5 — dominated by V1 on the rate side, and blocked by the container on the coefficient side

V5 (rank-8 learned basis: the top-8 principal directions of the 600 realized fields, pulled back to the
24×32 grid through the same mixing and re-quantised) was predicted at −4,070 B of basis plus −3,280 B of
coefficients. MEASURED: **−3,161 B total**, and the two halves fail for different reasons.

*Basis half.* Zeroing four of twelve atoms does not zero their cost: the shared 32-symbol Huffman table
gives symbol 0 a short but non-zero code, so 9,216 zero symbols still cost real bits. Measured basis
payload 10,945 B against 12,277 shipped — **−1,332 B**, not −4,070 B.

*Coefficient half — a hard container refusal, worth recording.* Zeroing coefficient columns 8–11 makes
their Rice parameter k = 0 while the live columns sit at k = 8–9. `ddm_up3.build_archive` refuses:
the packed CAP1 metadata stores k as **one bit per dimension over a u8 base**, so the twelve k values
must span at most 2 adjacent integers. Forcing the dropped columns to k = 8 to fit the field costs
600×(8+1) = 5,400 bits each instead of 600, wiping out 2,400 B of the 3,316 B the rank cut was supposed
to buy. **A rank cut cannot be spent inside this container without widening the packed k field.**

*Dominance.* V1 at optimal form buys −392 B at full rank; V5 buys −3,161 B but loses a third of the
carrier's dimensions. Both are below V2 (−3,204 B at full rank and full 3-bit fidelity), which keeps
every atom. V5 is therefore dominated by V2 on both axes simultaneously and is not solved.

**verdict_scope: FORMULATION — rank-8 SVD basis embedded in the shipped 12-slot CAP1 container.**
A rank cut with a widened packed-k field, or with `CARRIER_DIM` reduced in the receiver, is a different
object and is NOT closed by this row. See ITEM 3.

---

## 5b. V2 — the 3-bit basis is REFUSED, by the same lower-bound argument

V2 kept every atom and quantised at the fidelity-optimal per-atom step (per-atom cosine 0.970–0.992),
bought −3,204 B, and therefore had **+61.9%** of pose budget: break-even 9.931e-06 against a base of
6.134076e-06. That is the largest budget any full-rank variant on this board had.

**MEASURED, 48 pairs, full `jg5.refine_pair` re-solve, warm-started from the least-squares field
projection.** Every pair stopped at `no_improving_step` or `converged_below_materiality_floor` — again a
physical stop, not a budget. Solved-subset base mean 2.6029e-06; solved-subset variant mean 3.5609e-04.
**35 of 48 pairs got worse; 13 got better.**

    d_pose_n600  ≥  (sum of the 48 solved finals) / 600  =  0.01709236 / 600  =  2.8487e-05

against a break-even of 9.931e-06 — **2.87× past it** — and past ft1's ceiling (1.694e-05) by 1.68×.
As with V4, this is a strict lower bound over the whole population: unsolved pairs contribute
non-negatively and cannot pull the mean down. V2 is refused, and so is the V2+V3 composition that
inherits its basis.

The run was stopped at 48/600 once the bound crossed, and its CPU was moved to V3. Stopping a solve
whose admissibility is already decided is not a truncation of the verdict — the verdict is the bound,
and the bound was complete.

**verdict_scope: INSTANCE — the 3-bit alphabet at the fidelity-optimal per-atom step on this basis.**
Combined with V1's arithmetic refusal (§5), the honest family-level reading is in the exchange curve,
not in either bit depth: **on this carrier the basis payload cannot be cut without paying more in pose
than the bytes are worth, at either end of the quantiser-step curve.** The byte-greedy end (naive
`codes/2`, −4,071 B) destroys fidelity outright; the fidelity-greedy end (−392 B) barely saves anything;
and the middle point that actually had budget (3-bit optimal, −3,204 B) was measured and refused. What
is NOT closed: a step sweep between those points looking for a minimum of ΔS rather than an endpoint,
and basis families other than a re-quantisation of the shipped atoms.

---

## 7. V3 — the live row (IN FLIGHT, not a verdict)

V3 is the only variant still standing. It changes no basis byte: it multiplies the twelve coefficient
scales by 4 and re-solves the 600×12 lattice at that coarser step, which shrinks the AR1-predicted Rice
deltas. MEASURED rate: **−1,814 B = −1.20787e-03 S**, break-even d_pose **8.172e-06** (+33.2% over the
base). The n600 re-solve is running; **no verdict is claimed here until all 600 pairs are solved and the
archive is built.**

Interim state, recorded so this memo is honest at any moment it is read — explicitly NOT a verdict, and
NOT admissible as one (the solved set is a partial, order-biased subset; [[m88]]/[[m96]]):

    solved 87/600 · subset ratio (variant / base on the same pairs) 0.9798 · 66 better, 21 worse
    projection if every unsolved pair merely held its base value: 6.084e-06 vs break-even 8.172e-06

The direction is favourable and the margin is wide, which is why the run is being finished rather than
stopped early the way V2 was — V2's bound had already crossed, V3's has not.

**An attribution confound is open and is NOT resolved by this row (ITEM 4).** `refine_pair` at 40 outer
rounds is a more thorough solve than the one that produced the shipped codes, so a V3 pose gain may be
"more solving" rather than "the coarser lattice is free". That does not affect V3's ΔS against the
shipped body, which is what decides shipping. It does mean the control — a v0_base re-solve on the
shipped lattice — is owed, and that control is independently interesting: any d_pose it wins costs
**zero** bytes.

---

## 8. What this arm did not do

- No T4 row was fired. Modal was never dispatched; MAIN fires.
- No candidate archive was sealed unless §7 admits one.
- The V3 attribution confound is open and named in ITEM 4: `refine_pair` with 40 outer rounds is a more
  thorough solve than the one that produced the shipped codes, so part of any V3 pose gain may be
  "more solving" rather than "the coarser lattice is free". The ΔS against the shipped body is
  unaffected — that is the number that decides shipping — but the *attribution* is not established, and
  the control (a v0_base re-solve on the shipped lattice) is also the more interesting experiment: if it
  improves d_pose at **zero** bytes it is a free pointer move that belongs to nobody yet.

---

## ITEM 1 — `basis_scales` is 48 dead bytes; delete the field

MEASURED: all 12 per-atom basis scales are positive and cancel exactly in `normalized_basis`
(max abs difference 1.9073e-06, float32 epsilon on an RMS-1 field). The field costs 48 B in the carrier
body and reaches nothing. 48 B = 3.196e-05 S, which **clears the −2e-5 admit bar on its own**. Removing
it is a receiver format change (`cpr1/carrier_codec.py` prefix arithmetic plus `inflate.py:246`), i.e.
free code, and it is provably pose-neutral by construction rather than by measurement. Owner: unassigned.

## ITEM 2 — one more generated-basis attempt, matched to the Jacobian not the field

V4 is the largest prize on the board (−12,043 B). It failed with separable DCT bases whose span contains
2.49% of the realized field. The next attempt should not choose atoms by smoothness: it should choose a
*generic* atom family that maximises the smallest singular value of `d(pose)/d(coeff)` over pairs —
the quantity the solver actually inverts — and be re-scored the same way. Zernike, steerable pyramids,
and generated Gabor families are unmeasured. Falsifier and admissible window unchanged: n600 d_pose ≤
1.694e-05 (ft1 ceiling binds before break-even). Owner: unassigned.

## ITEM 3 — widen the packed CAP1 Rice-k field, then re-open the rank cut

The one-bit-per-dimension k field is what makes a rank cut unspendable (§6). Widening it to 2 bits costs
12 bits ≈ 2 B and would let dropped columns sit at k = 0. That turns V5's coefficient half from −916 B
(measured, after the k penalty) into roughly −3,316 B (DERIVED from the shipped 78,634-bit Rice payload
and the 4/12 column share) — to be MEASURED, not assumed. Owner: unassigned.

## ITEM 4 — re-solve the shipped lattice as the attribution control, and as a free-pointer probe

`jg5.refine_pair` at 40 outer rounds is a more thorough solve than the one that produced the shipped
codes. A v0_base n600 re-solve on the shipped basis and lattice (a) separates "coarser lattice is free"
from "more solving helps" for V3, and (b) is itself a candidate: any d_pose it wins costs **zero**
bytes, so its entire ΔS is negative by construction. Owner: unassigned.

---

## Frontier

    cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]
