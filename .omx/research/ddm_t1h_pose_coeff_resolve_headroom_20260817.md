# t1h — the zero-added-byte pose-coefficient headroom in the SHIPPED rr4 carrier

`date_utc: 2026-08-17` · `arm: ddm_t1h_pose_coeff_resolve_headroom_20260817`
`axis: [macOS-CPU advisory pose, EXACT local byte arithmetic, n600]`
`score_claim: false` · `promotable: false` · `pointer_moved: false`
payload: `/Volumes/APDataStore/pact/ddm_t1h/`

> # ⛔ REFUTED ON THE AUTHORITY AXIS — READ §11 BEFORE ANYTHING ELSE
> The pass-2 candidate was fired on contest-CUDA T4 (`fc-01M092BSTGSHRQAS2BJ1KCVARC`) and
> **REFUSED**. `d_seg` was unchanged and the byte arithmetic was exact — but `d_pose` **ROSE
> 6.31×** (6.886e-6 → 4.346e-5), taking S from 0.158534 to **0.171091 (+0.012557)**. The
> pre-registered falsifier 3 fired. **The CPU-torch accept oracle does not transfer to the
> contest axis; it ANTI-transfers.** Everything below §1–§10 that projects a lower S is
> superseded: those projections rest on a ratio-transfer assumption the row measured false.
> **Do not fire pass 2, pass 3, or the conservative variant.** §11 carries the verdict, the
> mechanism, the adjudication, and what is still standing.

## THE ANSWER, FIRST

**The headroom is large, and it is cheaper than free — but it is measured on an instrument
that reads this archive's pose 21.4× higher than the contest CUDA axis does, and that gap is
the whole risk.**

1. **Measured, n600, exact chain, two passes:** a single-coordinate integer re-solve of the
   shipped carrier codes takes CPU-torch `d_pose` from **1.4746613e-4 → 8.471492e-5**
   (pass 1, ratio 0.57447, 590/600 pairs improve) → **6.064679e-5** (pass 2, cumulative ratio
   **0.41126**, −58.87%, 523/600 still improving). Byte-identity control on the shipped
   lattice: **600/600, zero failures.** **The axis is NOT converged at pass 2.**
2. **The bytes barely move, in either direction.** Pass 1 costs **−5 B**; pass 2 costs
   **+8 B** on the counted archive (181,161 → 181,169 B). The packed carrier section is
   byte-length-identical (22,183 B in every case), so no offset moves.
3. **`d_seg` is invariant by construction**, not by hope: SegNet reads only the odd frame
   (`upstream/modules.py:108`) and no odd frame is touched.
4. **Two byte-closed candidates exist and every proof is green on both** — parse-back
   (receiver lattice matches intent, 0 mismatched coordinates), determinism repeat
   (byte-identical), and all five non-carrier sections byte-identical.
   **Pass 2 = `d2da8449…`, 181,169 B (primary). Pass 1 = `3dee2ee4…`, 181,156 B (fallback).**
5. **The headroom in S units.** Transferred to the T4 axis **as a ratio**:
   pass 1 `ΔS = −0.002011` → S 0.156522; **pass 2 `ΔS = −0.002970` → projected S 0.155563**.
   Against the charter's 1e-4 fork threshold this is **30×** over the bar, so this arm
   entered solve mode.
6. **The honest discount.** Our archive reads `d_pose` **1.4747e-4 on CPU-torch** and
   **6.88e-6 on contest-CUDA T4** — a **21.4×** level gap. So ~95% of the residual energy my
   accept oracle minimised is energy the CUDA axis does not see. The ratio-transfer in (5) is
   therefore an **upper** estimate; the pessimistic estimate is ≈ 0. I am not claiming
   −0.003. I am claiming a **bounded-downside T4 probe**: the byte delta is ±8 B and `d_seg`
   is invariant by construction, so the candidate **cannot lose** more than 5e-6 S, and one
   T4 row settles a question no amount of local work can.

   > ⛔ **THIS BOUND WAS WRONG, AND MEASURED WRONG: the row lost +0.012557 S, ~2,500× the
   > "cannot lose more than 5e-6" I wrote here.** The error is instructive and is dissected in
   > §11.2. In one line: I bounded the downside by treating "nothing transfers" as the floor,
   > while my own falsifier 3 explicitly anticipated `d_pose` RISING — the two claims
   > contradict each other and I shipped both in the same sealed order. Only `d_seg` and the
   > byte delta were ever bounded; the pose axis never was.

**Prediction trial:** this is the third discriminator alongside fx1's mixer and the QAT
continuation. The measured outcome — a large, cheap, structurally-explained gain from a
CONSTRAINED SOLVE over already-transmitted integers — **supports the constrained-solve
operator branch (alive subalgebra)**, not GS1-PRED. But the support is conditional on
transfer, and the T4 row is what converts it from support to evidence.

## 1. What PR133 actually did, and what the equivalent is on OUR vehicle

PR133's eval-bot-confirmed move (`0.172141 → 0.165780` [contest-CUDA T4]) decomposes, on the
author's own effort-matched control, into **89.5% a byte-frozen coefficient re-solve** and
10.5% their "CBQ" basis coarsening (hx1's intake,
`.omx/research/ddm_hx1_pr_wave_harvest_20260817.md`). The re-solve is a greedy coordinate
search over ALREADY-TRANSMITTED integer codes, accepted against an exact PoseNet forward,
with uint8 rounding inside the loop.

Our vehicle inherits the same carrier. Read at source in the shipped receiver:

| fact | source |
|---|---|
| the EVEN frame of each pair is rendered from that pair's 12 carrier codes alone | `cpr1/inflate.py:335-350` |
| the ODD frame comes from the semantic renderer + token stream, and carries no carrier code | `cpr1/inflate.py:315-330` |
| SegNet reads **only the last frame**: `x = x[:, -1, ...]` | `upstream/modules.py:108` |
| PoseNet reads the whole pair (interpolate to 384×512, **then** `rgb_to_yuv6`) | `upstream/modules.py:71-75` |
| the receiver lattice is `base_codes + compensation_overlay` | `runtime/f26_inflate.py:468-475` |

Two consequences follow, and both are structural rather than empirical:

1. **`d_seg` is invariant under any carrier-code move.** SegNet never sees the even frame.
   This is not a hope about a small perturbation; it is the scorer's own slicing.
2. **The 600 pairs are independent sub-problems.** Pair *i*'s pose energy is a function of
   pair *i*'s 12 codes only, so a per-pair coordinate search is exact and embarrassingly
   separable — and the per-pair winners compose into one candidate with no further scorer
   work.

## 2. Why headroom should exist here at all: 98.83% of the lattice was never treated

The shipped counted compensation overlay (`Q2C1`, 36 B) decodes to **7 pairs of 600
(1.17%)**, 30 changed coordinates, pair ids `[7, 96, 105, 176, 178, 517, 523]`. The overlay
format is hard-capped at 15 pairs (a 4-bit count field), so it *cannot* express more.

Everything else in the 600×12 lattice is inherited from the PR130/CPR1 carrier, fitted
against **their** odd frame. Our odd frame is a different renderer with a different token
stream. So for 593 of 600 pairs the shipped codes have never been evaluated against the
frame they are actually paired with on this vehicle.

## 3. The instrument, and its three controls

The measurement runs the EXACT shipped chain — hard `.round()`, the real sparse frame-0
selector, upstream's own PoseNet preprocess, frozen CPU-torch PoseNet. No straight-through
estimator, no linearization, no surrogate anywhere in the accept path. It reuses jc1's
chain (`experiments/ddm_jc1_carrier_pose_jacobian.py`) rather than opening a second
unverified one.

| control | what it proves | result |
|---|---|---|
| **byte-identity**: rendered frame_0 vs the retained `0.raw` even frame | the chain is the SHIPPED chain, selector and overlay included | see §5 |
| **CAP1 re-encode**: re-encode the shipped codes and compare to the shipped container | the byte pricer can reproduce what it prices | **exact**: 78,036 bits, ks `[9,9,9,8,8,9,9,9,9,9,9,9]`, 22,222 B — all match |
| **GT reuse**: re-derive ground-truth PoseNet rows from the video | jc1's retained GT (measured on a *different* archive) is legitimately reusable | **0.0 deviation** on 6 seeded random pairs (4, 55, 135, 287, 456, 590) |

The GT control matters because ground truth depends only on the upstream video and the
frozen PoseNet, never on our archive — but that is an argument, and an argument is not a
control. Decoded through upstream's own `yuv420_to_rgb` (never PyAV rgb24, which
manufactures ~100× phantom pose), the deviation is exactly zero, bit for bit.

**The render was not re-run.** rr4's own inflate output was located in retained custody at
`/Volumes/APDataStore/pact/ddm_rr2_encoder_build/parseback/inflated/0.raw`; its sha256
`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` equals
`RESULT_receiver_parseback.json:/inflated_output/sha256` for the pinned archive. Verified,
not assumed — rr4 is a rate-only re-encode whose carrier, compensation and model sections
are all byte-identical to its predecessor, so the render is the same bytes.

## 4. The byte price: a ±1 code move is FREE in this container

This is the half that decides whether any pose headroom is *spendable*, and it was measured
before any solve was attempted.

The carrier travels in the archive as `brotli(packed_CAP1 + overlay, quality=11, lgwin=24)`.
The counted stream is **22,161 B**; the body is 22,219 B = packed CAP1 (22,183) + overlay
(36). Brotli at the shipped parameters reproduces the shipped stream exactly, so the
container is reproducible before it is modified.

Inside CAP1 every field except the Rice residual payload is independent of the coefficient
codes, and the predictor is AR(1) — `restore_ar1_bias` predicts frame *i* from frame *i−1*
only. Inverting it, **one code move perturbs exactly two Rice symbols** (*i* and *i+1*).

Measured over 240 seeded random (pair, coord, ±1) positions:

| quantity | value |
|---|---|
| mean Δbits | **0.0** |
| median Δbits | 0 |
| range | −1 … +1 |
| fraction free or cheaper | **99.17%** |

The mechanism is Rice with k = 8–9: a ±1 change lands in the remainder bits, which are fixed
width, so the unary quotient is untouched unless the residual crosses a 2^k boundary. And
because the AR(1) inverse moves residual *i* and *i+1* in opposite directions, the two
changes routinely cancel.

**So "zero added bytes" is not an aspiration on this vehicle — it is the typical case.**

## 5. The headroom measurement

29,389 exact PoseNet forwards, 2,517 s, all 600 pairs, integer steps −2/−1/+1/+2 on each of
the 12 coordinates, every candidate evaluated through the hard-rounded shipped chain.

| quantity | value |
|---|---|
| base `d_pose` (CPU-torch, n600) | 1.4746613e-4 |
| single-coordinate optimal `d_pose` | 8.471492e-5 |
| ratio | **0.57447** |
| pairs improved | **590 / 600 (98.33%)** |
| byte-identity control | **600 / 600, 0 failures** |
| counted archive delta | **−5 B** |

The gain is broad, not a few lucky pairs — but it is not uniform either:

| top-N moves by absolute gain | share of total gain |
|---|---|
| 10 | 9.6% |
| 25 | 20.6% |
| 50 | 33.6% |
| 100 | 52.7% |
| 200 | 76.4% |
| 300 | 89.5% |

Half the gain lives in the top 100 pairs. That matters for the transfer risk in §6: the
tool exposes `--min-gain-frac`, which keeps only moves whose measured relative gain clears a
threshold. Small wins are the ones most likely to be instrument-specific, and this is the
knob that trades a little headroom for robustness. The shipped candidate keeps **all 590**
moves, because the byte cost is negative either way and `d_seg` cannot move.

### The pass curve (PR133's own honesty standard)

PR133 reported that neither its candidate nor its effort-matched control had converged by
pass 8. Ours has not converged either, and here is the curve so the claim is checkable:

| pass | `d_pose` (CPU-torch) | per-pass ratio | cumulative ratio | pairs improving | counted Δ bytes | T4 ratio-transfer S |
|---|---|---|---|---|---|---|
| shipped | 1.4746613e-4 | — | 1.00000 | — | 0 | 0.158533 |
| 1 | 8.471492e-5 | 0.57447 | 0.57447 | 590 / 600 | **−5** | 0.156522 |
| 2 | 6.064679e-5 | 0.71589 | **0.41126** | 523 / 600 | **+8** | **0.155563** |

Pass 2 still accepts on 87% of pairs, so pass 3 has not been ruled out — it has simply not
been run. Each pass costs ~30 min of local CPU and no money.

A small internal consistency check worth naming: pass 2's byte-identity counter reports
exactly **10** pairs matching the shipped render — precisely the 10 pairs pass 1 left
unmoved. The control is disabled off the shipped lattice, but its residual reading still
agrees with the bookkeeping.

### Why the shipped codes were so far from optimal

Not because anyone erred. The lattice is PR130/CPR1's, fitted against **their** odd frame;
ours is a different renderer over a different token stream. The only pose-directed treatment
our vehicle ever applied to it is the 36 B `Q2C1` overlay — **7 pairs, 1.17%**. The other
98.83% had never been evaluated against the frame they are actually scored with. The
measurement is best read as *finishing an inherited fit*, not as beating a solved problem.

### The instrument gap, stated plainly

`d_pose` on this exact archive: **1.4746613e-4** through the frozen CPU-torch chain (this arm,
n600) versus **6.88e-6** on the contest-CUDA T4 row.

The CPU figure is not an artefact of my code. jc1 measured 1.474678e-4 on the predecessor
lineage and the retained hv1 CPU auth-eval report prints 0.00014747 — three independent
paths agreeing to ~1.2e-5 relative. (They agree because rr4 is a rate-only re-encode: its
carrier, models and decoded tokens are byte-identical to that predecessor, so the render is
the same bytes — which is also why I could reuse the retained `0.raw` instead of re-rendering.)

That ratio, 21.4×, is not a rounding difference — and it is arithmetically forced, not
inferred: at `d_pose` = 1.4747e-4 the pose term alone would be 0.038402, which with the
0.120626 rate term already exceeds the measured S of 0.158533 and would require a negative
`d_seg`. So the T4 axis genuinely sees ~6.88e-6 and the CPU axis genuinely sees ~21× more.

The consequence for this arm is uncomfortable and must not be glossed: **the accept oracle
minimised an objective whose energy is ~95% invisible to the axis that scores us.** Two
readings are open and I cannot separate them locally — a PoseNet forward-numerics difference
(CPU vs CUDA kernels) or a ground-truth decode-path difference (the CPU run is tagged
`auth-eval env mismatch`; `AVVideoDataset` vs `DaliVideoDataset`). Either way the selected
moves may transfer fully, partially, or not at all. That is precisely why the candidate is
built so that a failed transfer costs nothing.

## 6. What is NOT established

- **This is not a score.** The accept oracle is CPU-torch PoseNet, the only pose authority
  this arm may run. Only `upstream/evaluate.py` on contest hardware produces a score.
- **The CPU→CUDA transfer of the *improvement* is an assumption, not a measurement.** The
  base agreement between the two instruments is reported below, which is evidence about the
  operating point, not proof that a selected edit survives the change of instrument. A T4
  row is the only thing that settles it, and the falsifiers are pre-registered for exactly
  that reason.
- **No archive was byte-closed by this arm.** The counted price is computed by rebuilding
  the carrier body and re-compressing it with the shipped brotli parameters; a shippable
  candidate additionally needs the container writer path (§8).

## 7. STORES CONSULTED

- `.omx/research/ddm_hx1_pr_wave_harvest_20260817.md` + `/Volumes/APDataStore/pact/ddm_hx1/notes/pr133_pr132_notes.md`
  — the PR133 mechanism, its 89.5%/10.5% split, and the falsifier that "matched8" is an
  effort control, not an 8-bit quantizer.
- `.omx/research/ddm_jc1_carrier_jacobian_posemetric_refit_20260816.md` — K = 0 (no free
  carrier direction), cond(J_stack) = 12.02, and the warning that first-order carrier
  promises were wrong by 134–1,065× when measured exactly. That is why this arm's accept
  oracle is the exact chain and never the Jacobian.
- `.omx/research/ddm_me1_micro_edit_engine_20260817.md` — the gap is 97.20% pose; zeroing
  `d_pose` leaves 0.00023867 S at the rr4 base. Source of the T4 `d_pose` = 6.88e-6.
- `.omx/research/ddm_qs1_frame0_schur_coupled_solve_20260813.md` — exact frame-0 signed-int12
  moves cancelled up to 99.995% of a frame-1 edit's PoseNet-6 leakage; the frame-0 code
  lattice is a proven strong pose actuator on this vehicle.
- `.omx/research/ddm_qs2_compensation_rate_rung_20260813.md` — the counted `Q2C1` overlay
  this arm decoded and left byte-identical.
- `experiments/ddm_ra2b_carrier_chain_control.py` — the proven decode chain (its header
  records that an earlier arm SKIPPED the selector split and the overlay; both are honoured
  here).
- `.omx/state/canonical_frontier_pointer.json` — rr4 sha `35ac2b9b…`, 181,161 B, S 0.15853325
  [contest-CUDA T4].

## 8. SEALED FIRE-ORDER (for MAIN) — ⛔ FIRED AND REFUSED, DO NOT RE-FIRE

> The pass-2 primary below WAS fired (`fc-01M092BSTGSHRQAS2BJ1KCVARC`, 454 s) and REFUSED:
> `d_pose` rose 6.31×, S 0.171091. The projected-S rows in this section are superseded by
> §11. The candidate bytes remain retained as the negative's evidence, not as a fire-order.

**PRIMARY candidate — pass 2 (t1h-on-rr4):**

| field | value |
|---|---|
| archive | `/Volumes/APDataStore/pact/ddm_t1h/candidate_pass2/archive.zip` |
| sha256 | `d2da8449420be1a22d7c4a1799c2530062a2b3b448cb5d3fb9b836864a7fe754` |
| bytes | **181,169** (shipped 181,161; **+8 B**) |
| determinism repeat | `archive.repeat.zip`, byte-identical |
| projected S (ratio transfer) | 0.155563 |

**FALLBACK candidate — pass 1** (smaller, more conservative, fewer selected moves):

| field | value |
|---|---|
| archive | `/Volumes/APDataStore/pact/ddm_t1h/candidate/archive.zip` |
| sha256 | `3dee2ee4c9ed7ee81f5221b64258b43802a3d32c5ae08a193c707dae693dd0b3` |
| bytes | **181,156** (**−5 B**) |
| projected S (ratio transfer) | 0.156522 |

Both share: base rr4 `35ac2b9b…` 181,161 B; **runtime unchanged** — the receiver is not
modified in any way.

**Standalone swappable section (for fx1's mixer byte-close):**

| field | pass 2 (primary) | pass 1 (fallback) |
|---|---|---|
| path | `candidate_pass2/carrier_section_candidate.bin` | `candidate/carrier_section_candidate.bin` |
| sha256 | `8ddeeb42dcf532f5e3f56bdabe0b0010966541303ab3a071b1160c0e75480b81` | `ab0a2e61ec5bc925f8be4578a6885fbf0f543cf0ea1b135c54672d55f6550e1f` |
| bytes | **22,183** | **22,183** |

Shipped section for diff: `carrier_section_shipped.bin`, sha `30c33886dcf40684…`, 22,183 B.
**All three are the same length**, so the section is a true drop-in.

It is the packed CAP1 carrier section, which the receiver dispatches on by exact length, so
dropping it into another byte-close moves no offset. Only the Rice payload and its two
header fields differ; the basis bitstream, the 9-byte frame-0 selector suffix, the AR(1)
metadata and the Rice parameters are all carried through unchanged.

**Pre-registered falsifiers for the T4 row (all three must hold):**

1. **`d_seg` UNCHANGED to every printed digit.** Structural: no odd frame is touched. If
   `d_seg` moves at all, the build is wrong, not the theory — stop and investigate.
2. **Archive bytes EXACTLY 181,169** (pass 2) or **181,156** (pass 1).
3. **`d_pose` FALLS.** If it rises, the CPU-torch accept oracle does not select moves that
   survive the CUDA instrument, and this whole family is refuted on this vehicle. That is a
   clean, valuable negative and should be recorded as one.

**Expected band on the pose axis, pass 2.** Optimistic (ratio transfers): `d_pose`
6.88e-6 → 2.83e-6, S → 0.155563. Pessimistic (nothing transfers): `d_pose` unchanged,
S → 0.158539 (the +8 B alone). **Anything in that band is a win or a ~5e-6 tie; only a
`d_pose` INCREASE is a real loss.**

**If only one row can be bought, buy pass 2.** It dominates pass 1 under any positive
transfer, and its worst case is 13 B worse.

## 9. NEXT_IF_RESUMED

1. **Pass 3.** Pass 2 still accepted on 523/600 pairs, so the axis is not exhausted. One
   command, ~30 min, no money:
   `ddm_t1h_pose_coeff_headroom.py --pairs all --deltas=-2,-1,1,2 --start-codes
   pass2/candidate_receiver_codes.int32.npy --receipt T1H_HEADROOM_PASS3.json`, then compose
   with `--base-codes pass2/candidate_base_codes.int32.npy` and rebuild. Note the compose
   receipt's `t4_ratio_transfer` is PER-PASS; the cumulative ratio must be taken against the
   shipped base (1.4746613e-4), as §5's table does.
2. **Multi-coordinate moves per pair are unmeasured.** Only one coordinate per pair was
   swept. The per-pair problem is 12-dimensional and jc1 measured cond(J_stack) = 12.02 with
   no null direction, so joint moves should buy more than the sum of single ones.
3. **The instrument question is the real blocker, and it is worth one dedicated row.** If a
   CUDA-side per-pair `pose6` dump is ever cheap to obtain, the winner's-curse discount
   becomes measurable rather than argued, and this whole family becomes schedulable instead
   of speculative.
4. **The `--min-gain-frac` conservative variant is built but unfired.** If the T4 row shows
   partial transfer, re-run the compose with a threshold and keep only the top moves.

---

# 10. PASS 3 — and the channel's measured capacity (appended 2026-08-17, respawn)

> ⛔ **SUPERSEDED BY §11 for every S projection and the fire-order in §10.5.** Pass 3 composes
> MORE of the same CPU-oracle moves that the T4 row refuted, so it is **mooted as filed and
> must not be fired.** What SURVIVES from this section is the apparatus and the structural
> measurement: the 78,040-bit container ceiling, the bit-budget fit, the byte-cost trend, and
> the reach table — all of which are oracle-independent and are reused in §11.

## The answer, first

**Pass 3 lands, but it also found the wall: the zero-added-byte channel has a HARD, measured
capacity of 78,040 Rice bits, and pass 2 had already consumed the last 4 bits of it.** Pass 3's
unconstrained composition needs 78,042 — two bits too many — and the pricer REFUSED it rather
than mis-price a body the container cannot carry. Fitting it back inside costs almost nothing
(2 substitutions, 2.6e-7 of energy, **99.9994%** of the gain retained), so pass 3 is real; but
the "free byte" story from §4 is now qualified, and the qualification is structural.

| pass | `d_pose` (CPU-torch) | cumulative ratio | counted Δ B | S (ratio transfers) | S (nothing transfers) |
|---|---|---|---|---|---|
| shipped rr4 | 1.4746613e-4 | 1.00000 | 0 | 0.158533 | 0.158533 |
| 1 | 8.471492e-5 | 0.57447 | −5 | 0.156522 | 0.158530 |
| 2 (**FIRED**) | 6.0646787e-5 | 0.41126 | +8 | 0.155563 | 0.158539 |
| 3 conservative | 5.3144449e-5 | 0.36038 | +10 | 0.155225 | 0.158540 |
| **3 full** | **4.8105804e-5** | **0.32622** | **+22** | **0.154991** | 0.158548 |

## 10.1 The container has a hard bit budget, and it is now saturated

The packed CAP1 section is dispatched by the receiver on an EXACT length, so the Rice payload
field inside it has a fixed byte width. Read from the shipped archive rather than assumed:

| quantity | value |
|---|---|
| shipped Rice bits | 78,036 |
| payload field width | 9,755 B |
| **maximum Rice bits the container can carry** | **78,040** |
| shipped slack | **4 bits** |
| pass 2 Rice bits | **78,040 — exactly at the ceiling** |
| pass 3 unconstrained | 78,042 (**+2 over**) |

So pass 2 did not merely cost +8 B; it spent the container's entire remaining bit slack. That
is the fact §4's "a ±1 code move is FREE" needed and did not have: the move is free *per move*,
but the aggregate has a ceiling, and three passes reached it.

**The first compose attempt failed closed** — `COUNTED PRICE UNAVAILABLE: candidate Rice payload
is 9756 B but the packed section is dispatched on an exact length requiring 9755 B`. That refusal
is the tool working. It is also why this section exists rather than a quietly over-long archive.

## 10.2 Fitting inside the budget: solve it, do not truncate it

The obvious repair — drop the smallest-gain moves — is terrible here. Measured:

| moves kept (by gain rank) | Rice bits | fits? | gain kept |
|---|---|---|---|
| 428 (all) | 78,042 | no | 100% |
| 300 | 78,042 | no | 99.95% |
| 200 | 78,041 | no | 98.2% |
| 100 | 78,040 | **yes** | **81.1%** |

Shedding 2 bits by gain-rank costs 19% of the gain, because gain rank and bit cost are almost
unrelated. The sweep already stores every coordinate's exact energy (`per_coord`), and pairs are
independent, so the right move is a constrained solve over already-measured quantities: **maximise
measured pose gain subject to `rice_bits ≤ 78,040`**, substituting among measured options.

`fit_to_bit_budget` (landed this session, commit `aa716795ba`) does exactly that — exhaustive over
SINGLE substitutions (every pair × every measured per-coordinate option, plus reverting), choosing
the minimum `energy increase / bits saved`. Two substitutions sufficed:

| pair | substituted to | bits saved | energy cost |
|---|---|---|---|
| 443 | coord 4, δ +2 | 1 | 7.46e-8 |
| 110 | coord 0, δ +2 | 1 | 1.88e-7 |

Total cost **2.62e-7** against a 4.51e-2 gain: **99.9994% retained**, versus 81.1% for the
truncation. No new scorer forward was run — every energy above was already measured exactly.

Joint substitutions are not searched, so this is a good feasible point, not a proven optimum.

## 10.3 The byte cost is rising, and it is not monotone in move count

−5 B (pass 1) → +8 B (pass 2) → +22 B (pass 3). Re-solving moves codes off the AR(1)-predictable
manifold, so the residuals get less brotli-compressible with each pass. The counted cost is also
**not monotone in the number of moves** — fewer moves is not reliably fewer bytes:

| `--min-gain-frac` | moves kept | `d_pose` | cumulative | counted Δ B | S (ratio transfers) |
|---|---|---|---|---|---|
| 0.00 (full) | 428 | 4.810580e-5 | 0.32622 | +22 | 0.154991 |
| 0.05 | 276 | 4.828310e-5 | 0.32742 | +17 | 0.154996 |
| 0.10 | 244 | 4.865545e-5 | 0.32994 | +35 | 0.155026 |
| 0.20 | 185 | 5.114747e-5 | 0.34684 | +30 | 0.155144 |
| **0.30** | **154** | **5.314445e-5** | **0.36038** | **+10** | **0.155225** |

Every one of these still overflowed by 1 bit before repair — the ceiling binds regardless of
threshold. And thr 0.30 is *cheaper in bytes* than thr 0.10 while keeping fewer, higher-confidence
moves, which is why it is the conservative rung rather than 0.20.

## 10.4 What the threshold ladder can and cannot answer

A correctness constraint that shapes this: the composition identity ("candidate `d_pose` = mean of
the measured per-pair energies") holds only when moves are folded into the lattice the sweep
measured from. Pairs are independent, so thresholding a **later** pass on top of a **full** earlier
pass is exactly measurable — but thresholding an **earlier** pass invalidates the later sweeps'
energies for every pair whose earlier move was dropped. So the ladder above is a threshold on the
pass-3 layer only, and a cumulative-thresholded chain would need the sweeps re-run.

The data also weakens the winner's-curse worry for the early passes: at a 20% relative-gain
threshold **536 of pass 1's 590 moves survive**, holding 96.7% of its gain. Pass-1 moves typically
cut a pair's pose energy by well over 20% — they are not noise-scale wins. The marginal moves live
in pass 3, which is precisely where the threshold is applied.

## 10.5 SEALED FIRE-ORDER — pass 3 (for MAIN) — ⛔ WITHDRAWN, DO NOT FIRE

> Withdrawn 2026-08-17 on the pass-2 T4 refusal. Both candidates below are byte-valid and
> every structural proof still holds; what fails is the ORACLE that selected their moves. The
> shas are retained so the negative is reproducible, not so the archives can be shipped.

**PRIMARY — pass 3 full:**

| field | value |
|---|---|
| archive | `/Volumes/APDataStore/pact/ddm_t1h/candidate_pass3/archive.zip` |
| sha256 | `bbb7c3650f890c8449762c65911777c0b6e1a58f322857ad1eb5845226f11534` |
| bytes | **181,183** (shipped 181,161; **+22 B**) |
| determinism repeat | byte-identical |
| parse-back | receiver lattice matches intent, **0** mismatched coordinates |
| non-carrier sections | all five byte-identical |
| projected S (cumulative ratio transfer) | **0.154991** |

**CONSERVATIVE — pass 3 at `--min-gain-frac 0.30`** (fewer, higher-confidence moves, cheaper):

| field | value |
|---|---|
| archive | `/Volumes/APDataStore/pact/ddm_t1h/candidate_pass3_conservative/archive.zip` |
| sha256 | `b6ebb65a78e89fc206d0e6bf4c43c959c29b04ea24aa87b939564f1d34786c3e` |
| bytes | **181,171** (**+10 B**) |
| projected S (cumulative ratio transfer) | **0.155225** |

Both: base rr4 `35ac2b9b…`; **runtime unchanged**; all proofs green.

**Standalone swappable sections for fx1** — both byte-LENGTH-identical to shipped (22,183 B), so
the drop-in contract of §8 still holds and no offset moves:

| variant | path | sha256 |
|---|---|---|
| pass 3 full | `candidate_pass3/carrier_section_candidate.bin` | `cacfe9cd24259e730f05cc9244a9bc3d78015186f29edb0c0d0cc26df036a172` |
| pass 3 conservative | `candidate_pass3_conservative/carrier_section_candidate.bin` | `25c1e19ee37b9cefb7e0cc87a1c7b98bb5f592b121e4995cd7fa5745f78ee4c9` |
| shipped (for diff) | `carrier_section_shipped.bin` | `30c33886dcf40684a5895c48e292d11a9180380f9d1219c0c6de81754bbb3aab` |

⚠ **New constraint fx1 must honour:** the section is at its Rice-bit ceiling (78,040 / 78,040). A
mixer that composes this section with any other carrier edit has **zero bits of slack** and must
run the container-fit repair, or it will produce an unpackable section.

**Pre-registered falsifiers — unchanged in kind, updated in value (all three must hold):**

1. **`d_seg` UNCHANGED to every printed digit.** Structural: no odd frame is touched.
2. **Archive bytes EXACTLY 181,183** (full) or **181,171** (conservative).
3. **`d_pose` FALLS.** If it rises, the CPU-torch accept oracle does not select moves that survive
   the CUDA instrument, and this family is refuted on this vehicle — a clean, valuable negative.

**Expected band, pass 3 full.** Optimistic (cumulative ratio transfers): `d_pose` 6.88e-6 →
**2.24e-6**, S → 0.154991. Pessimistic (nothing transfers): S → 0.158548, a **1.5e-5** loss from
the +22 B alone. Only a `d_pose` increase is a real loss.

**Ordering advice.** Do not buy a pass-3 row until the pass-2 row returns: pass 2 and pass 3 test
the SAME transfer hypothesis, and pass 2's result re-prices pass 3 exactly. If pass 2 transfers,
fire pass 3 full. If pass 2 transfers only partially, fire pass 3 conservative. If pass 2 shows no
transfer or a rise, fire neither — thresholds keep the same *kind* of move and cannot rescue a
failed transfer.

## 10.6 Controls run this session

| control | result |
|---|---|
| pass-2 recompose under the refactored selector | lattice **byte-identical**, receipt numbers identical (`pass2_regression/`) |
| pass-3 recompose after the review fixes | lattice **byte-identical** (`pass3_recheck/`) |
| conservative archive rebuilt from the regenerated lattice | **byte-identical**, not stale (`candidate_pass3_conservative_recheck/`) |
| determinism repeat, both new candidates | byte-identical |
| parse-back, both new candidates | 0 mismatched coordinates |
| int12 guard mutation control | removing the guard makes the repair emit **2049**, outside signed-int12 — the test catches it |
| new unit tests | 5 passed (`src/tac/tests/test_ddm_t1h_container_fit.py`) |

## 10.7 NEXT_IF_RESUMED (supersedes §9 items 1 and 4)

1. **Pass 4 is now a bad trade, and that is a measured verdict, not a guess.** The container is
   saturated, so every further pass must BUY its bits by giving back pose, and the byte trend
   (−5 → +8 → +22) is rising. Pass 4 would be the first pass whose repair cost is likely to be
   material. Do not run it before a T4 row settles whether ANY of this transfers.
2. **Multi-coordinate moves per pair remain unmeasured** (§9 item 2 stands, unchanged). They are
   now MORE attractive than another pass: jc1 measured cond(J_stack) = 12.02 with no null
   direction, so a joint per-pair solve should buy more pose per bit than a fourth single-coordinate
   sweep — and bits, not forwards, are now the scarce resource.
3. **The instrument question is still the real blocker** (§9 item 3, unchanged and now more urgent:
   three passes of local gain are staked on one untested transfer assumption).
4. **If the container ceiling itself is worth attacking**, the lever is the packed section's
   exact-length dispatch, not the codes — that is a receiver/format change and belongs with fx1,
   not inside this arm's "runtime unchanged" contract.

---

# 11. THE T4 VERDICT — the accept oracle ANTI-transfers (appended 2026-08-17)

## The answer, first

**Fired, refused, and the refusal is more informative than a pass would have been.** On
contest-CUDA T4 (`fc-01M092BSTGSHRQAS2BJ1KCVARC`, `passed=true`, 454 s) the pass-2 candidate
scored **S 0.171091 against rr4's 0.158534 — a loss of +0.012557.** Two of my three
pre-registered falsifiers CONFIRMED the apparatus; the third refuted the science:

| falsifier | prediction | measured on T4 | verdict |
|---|---|---|---|
| 1. `d_seg` unchanged to every digit | structural (SegNet reads only the odd frame) | seg contribution **0.029611**, unchanged | **PASS** |
| 2. archive bytes exactly 181,169 | exact local byte arithmetic | **181,169** | **PASS** |
| 3. `d_pose` FALLS | 6.886e-6 → 2.83e-6 | **4.346e-5 — ROSE 6.31×** | **FIRED** |

I wrote of falsifier 3: *"If it rises, the CPU-torch accept oracle does not select moves that
survive the CUDA instrument, and this whole family is refuted on this vehicle. That is a clean,
valuable negative and should be recorded as one."* It rose. Recording it as one.

## 11.1 The mechanism: the oracle spent real energy to buy phantom energy

The two instruments did not merely disagree in level — **they moved in opposite directions
under the same edit**:

| quantity | shipped rr4 | pass 2 | change |
|---|---|---|---|
| CPU-torch `d_pose` (the accept oracle) | 1.4747e-4 | 6.0647e-5 | **fell 2.43×** |
| contest-CUDA `d_pose` (the authority) | 6.886e-6 | 4.346e-5 | **ROSE 6.31×** |
| CPU / CUDA level ratio | 21.42× | **1.40×** | collapsed 15.3× |

That collapse kills the model §5 was working under. §5 treated the 21.4× as a fixed level gap
and transferred the improvement **as a ratio**; a fixed gap cannot collapse to 1.40× under an
edit. The gap was never a scale factor — it was a quantity the optimizer could and did consume.

Bookkeeping this as `CPU = shared + phantom`, with CUDA measuring the shared part (**a MODEL,
not a measurement** — the two chains may simply compute different functions, and additivity is
assumed, not shown):

| component | shipped | pass 2 | change |
|---|---|---|---|
| phantom (CPU-only) | 1.4058e-4 | 1.7187e-5 | fell 8.18× |
| shared (CUDA-visible) | 6.886e-6 | 4.346e-5 | rose 6.31× |

**The oracle removed 1.234e-4 of phantom energy and paid 3.657e-5 of REAL energy for it — an
exchange rate of 0.296 real lost per phantom removed.** It is not that the moves failed to
transfer. It is that the objective was ~95% phantom (§5 said this), so the argmin walked in a
direction that was free in the phantom coordinate and expensive in the real one. Under that
mechanism more passes dig deeper into the trade, which is why **pass 3 is predicted worse, not
better** — direction predicted by the mechanism, magnitude NOT predicted from a single row.

## 11.2 My own error, plainly

I shipped two contradictory claims in one sealed order. §1 item 6 said the candidate *"cannot
lose more than 5e-6 S"*; falsifier 3 said *"if `d_pose` rises, the family is refuted."* **A
pre-registered falsifier for event X is an admission that X is possible, so no downside bound
may assume X cannot happen.** The real bound was: `d_seg` bounded (structural), rate bounded
(±8 B, exact), **pose unbounded**. The measured loss was ~2,500× my stated worst case.

The lesson generalizes past this arm: *a "bounded-downside probe" is only bounded on the axes
whose bound is structural or arithmetic. An axis governed by an untested transfer assumption is
unbounded, and calling the probe "bounded-downside" launders that assumption into a guarantee.*
This is the same genus as the banked "solve objective must live on the shipping axis" law and
the "surrogate ≠ authority" NO-FAKE class; it composes with #1054's 21× CPU-pose degradation on
identical bytes into one fact: **pose acceptance must be measured on the shipping axis.**

## 11.3 Adjudicating the three reformulations

### (a) Dual-instrument acceptance (CPU-torch ∧ MLX-PoseNet) — **REJECT as filed**

The failure mode is not instrument NOISE, which a two-instrument intersection would suppress. It
is a systematic **ordering** disagreement worth 6.31× on the authority axis. MLX-PoseNet is a
third OFF-AXIS instrument, and the decisive question is where the phantom originates:

- **If the phantom lives in the GROUND TRUTH** (our chain decodes GT through upstream's
  `yuv420_to_rgb`; the contest axis uses `DaliVideoDataset`), then MLX inherits the identical GT
  and the identical phantom. The intersection is then no closer to CUDA and the whole scheme is
  a no-op dressed as a control.
- **If the phantom lives in the FORWARD** (CPU vs CUDA PoseNet kernels), MLX is a genuinely
  different forward and the intersection could help.

**I cannot currently tell which**, and — this is a blind spot in my own §3 controls that I
should name — my GT control proved *self-consistency*, not cross-instrument agreement: it
compared jc1's retained GT against a re-derivation through the **same** upstream decode. It
never touched the Dali path. So it could not have caught a GT-origin phantom.

Also: the banked 0.55% MLX drift law is a **LEVEL** agreement law. Nothing in it constrains
per-move ORDERING, which is precisely the quantity that failed here. Invoking it to license an
ordering-sensitive accept rule would be a cross-regime constant transfer — a named failure genus.

**Verdict: blocked on a precondition, not adopted.** Localize the phantom to FORWARD vs GT
first (§11.4 item 1). If it is GT-origin, (a) is dead on arrival.

### (b) Finer CUDA-side granularity — **honest split answer**

Per-**pass** CUDA verification is not unaffordable: it is exactly what we just did, it cost one
row and 454 s, and it **worked** — it caught a candidate that would have cost +0.0126 S before
it shipped. That is the gate earning its keep.

Per-**move** CUDA acceptance in the naive form IS unaffordable, and I will say so plainly: the
pass-3 sweep alone ran **29,390 exact forwards**, and moving that on-axis means running the
eval-side chain (Dali GT + CUDA PoseNet) 29,390 times per pass. **No cheap per-move on-axis
oracle exists today.**

But there is a cheap middle rung that does not exist only because nobody retained it. **The
scorer already computes 600 per-pair pose vectors on every single eval and reduces them to one
scalar.** Retaining that `600×6` array costs **~14 KB and zero extra compute**. With it from the
two rows ALREADY fired (rr4 and pass-2), the per-pair correlation between CPU-predicted gain and
CUDA-realised delta becomes measurable **for free**, and that one map decides the question §11.3(c)
otherwise has to close on argument:

- if the correlation is positive for large moves and negative only in the tail → `--min-gain-frac`
  is revived, with the threshold set on CUDA-side evidence instead of CPU-side faith;
- if the anti-correlation is uniform across the gain range → the family closes cleanly.

**This is the highest-value, cheapest next step in the whole arm, and it is APPARATUS, not a new
solve.** That the array was discarded is a measure-and-discard signal loss in the eval harness,
not a law of nature.

### (c) Formulation-scoped close — **ACCEPT, with the scope stated exactly**

**REFUTED (verdict_scope: FORMULATION):** *accepting carrier-code moves against the frozen
CPU-torch PoseNet chain on this vehicle.* Evidence: n=1 contest-CUDA row, pre-registered
falsifier, effect size 6.31× — decisive at this scope despite n=1 because the predicted sign was
wrong, not merely the magnitude.

**NOT refuted, and explicitly preserved:**

1. **The carrier re-solve PARADIGM.** PR133's eval-bot-confirmed `0.172141 → 0.165780` is the
   same mechanism working — with its accept oracle **on its scoring axis**. Our defect is the
   oracle, not the idea. Per CLAUDE.md "Forbidden premature KILL", this is
   **DEFERRED-pending-on-axis-oracle**, not KILL. Reactivation criterion: an accept oracle whose
   objective is measured on the shipping axis (§11.4 items 1–2).
2. **`d_seg` invariance under carrier moves — now CONFIRMED ON THE AUTHORITY AXIS.** This was a
   structural argument in §1; it is now a measured fact on contest CUDA, to every printed digit,
   under 1,113 changed coordinates. **The carrier is a pure-pose actuator with zero seg risk.**
   That is a genuine positive result and it is reusable by fx1 and any future carrier work.
3. **The byte-pricing and container apparatus — CONFIRMED ON THE AUTHORITY AXIS.** Predicted
   181,169 B, measured 181,169 B. The CAP1 re-encode, the brotli pricing, the parse-back and the
   78,040-bit ceiling of §10.1 are all oracle-independent and all stand.
4. **The bit-budget fit** (`fit_to_bit_budget`, commit `aa716795ba`) is a constrained solve over
   measured quantities. It is orthogonal to which oracle supplies the energies and will be needed
   by any successor that re-solves this carrier.

## 11.4 NEXT_IF_RESUMED (supersedes §10.7)

1. **Retain the per-pair `600×6` pose array in the eval harness, then re-read the two rows
   already fired.** ~14 KB, zero extra compute, and it converts this arm's scalar negative into a
   per-pair map that either revives the family with a CUDA-side threshold or closes it. **Do this
   before any further solve.** If the fired rows' artifacts happen to have retained per-pair
   components already, the analysis is free today.
2. **Localize the phantom: FORWARD or GROUND TRUTH?** The single diagnostic is our GT versus the
   contest's `DaliVideoDataset` GT on the same frames. This gates reformulation (a) entirely and
   explains the 21.4× level gap that has now cost one row. It also repairs the blind spot in §3's
   control.
3. **A random-perturbation control would separate two readings I currently cannot.** Is `d_pose`
   up 6.31× because the moves are ANTI-selected, or because ANY perturbation of this support size
   costs ~6× (i.e. the carrier already sits at a sharp CUDA optimum)? These imply opposite
   futures — a better oracle wins in the first, nothing wins in the second. One T4 row on a
   seeded random ±1 perturbation over the same 1,113 coordinates decides it. **Cost: one row.
   MAIN owns whether that is worth buying; I am not sealing it.**
4. **Multi-coordinate per-pair moves stay QUEUED but are now BLOCKED behind item 1.** A richer
   move set searched against a refuted oracle would only find a better phantom.
5. **Do not fire pass 2, pass 3, or the conservative variant.** All four candidate archives stay
   retained as the negative's evidence (shas in §8 and §10.5).

## 11.5 STORES CONSULTED (this section)

- The T4 row `fc-01M092BSTGSHRQAS2BJ1KCVARC` as relayed by MAIN: seg 0.029611, pose 0.020847062143141415,
  bytes 181,169, S ≈ 0.171091. Component arithmetic re-derived here rather than accepted:
  `d_pose = pose_contribution² / 10 = 4.346e-5`; `S = 0.029611 + 0.020847 + 25·181169/37,545,489
  = 0.171091`. ✓
- `#1054` — 21× CPU-pose degradation on identical bytes; the sister observation this composes with.
- §5 of this memo — the 21.4× instrument gap and its two open readings, now the direct cause of
  the refusal.
- `.omx/research/ddm_hx1_pr_wave_harvest_20260817.md` — PR133's re-solve worked with an on-axis
  accept oracle, which is what isolates OUR defect to the oracle rather than the paradigm.
