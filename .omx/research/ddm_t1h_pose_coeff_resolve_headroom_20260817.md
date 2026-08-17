# t1h — the zero-added-byte pose-coefficient headroom in the SHIPPED rr4 carrier

`date_utc: 2026-08-17` · `arm: ddm_t1h_pose_coeff_resolve_headroom_20260817`
`axis: [macOS-CPU advisory pose, EXACT local byte arithmetic, n600]`
`score_claim: false` · `promotable: false` · `pointer_moved: false`
payload: `/Volumes/APDataStore/pact/ddm_t1h/`

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

## 8. SEALED FIRE-ORDER (for MAIN)

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
