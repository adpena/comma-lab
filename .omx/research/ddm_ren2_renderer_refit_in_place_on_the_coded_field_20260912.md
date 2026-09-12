# ddm_ren2 — the renderer refit-in-place, RUN on the coded field: the init restores EXACTLY, the carrier does absorb a real refit, and the refit is not better than a coin flip of its own size

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · promotion_eligible=false ·
pointer_moved=false

<!-- # FORMALIZATION_PENDING: the three numbers this arm proposes as laws -- the refit AMPLITUDE
CURVE (a real refit's render delta is a sparse +/-1 dither, noise-dominated at small amplitude, that
loads onto the smooth band monotonically as it grows), the SURROGATE INVERSION (the expected-flip
objective descends while realized d_seg rises, monotonically, on two independent trajectories), and
the RANDOM-DIRECTION PARITY (the trained refit pays the same d_seg as an amplitude-matched random
dither) -- are n=1 on ONE object with two trajectories. They are registered as canonical equations
only when a second object reproduces them, per the binding-numbers-expire discipline; this memo says
so rather than promoting them. The member-container correction and the depth-table correction in §6
are BYTE-IDENTITY receipts, not models, and bind immediately. -->

Base: move 48, S 0.13638261682704697 @ 179,111 B `[contest-CUDA T4 n600]`, archive
`d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`; components d_seg 0.00010345,
d_pose 4.59e-06; admit bar −2e-5.

---

## ANSWER FIRST

**The pointer did not move. No checkpoint cleared the bar, nothing is staged, MAIN fires nothing.**
The refit ran — correctly instrumented for the first time in this campaign — and the axis closed on
SEG at every amplitude the carrier can absorb.

**The gate's first leg PASSED perfectly and that is the arm's most reusable result.** The restored
init re-encodes to the shipped member **byte-for-byte at all three stages** (SM3R body 36,130 B
`17e0fd0b…`, SM1S rider 31,451 B `9c0b579c…`, RX1M member 29,862 B `786950a5…`), renders
**bit-identically** to the shipped renderer over 120 seeded-random pairs, and reproduces `ddm_ren1`'s
n600 control to **eleven significant figures** on seg (`0.00010338677300347222`) and seven on pose
(`4.586844968e-06`). The member pricer is therefore the shipped encoder, not a look-alike, and the
0-byte / 0-LSB / 0-cell control row proves it.

**Leg 2 — the refit itself — is a clean negative, and the strongest single number is this:**

> At the SMALLEST amplitude the deployed grid can express (**0.1427 LSB camera RMS**, after 25
> optimizer steps), the refit moves **312 argmax cells and damages 308 of them** — **98.7 % of every
> cell it touches, it moves from correct to wrong** — and **0 of 600 pairs improve.** `d_seg` rises
> **2.53 %** where it had to FALL ~1 %. Not one pair, at any amplitude tested.

**And the falsifier that makes it a finding rather than a disappointment:** an amplitude-MATCHED
random ±1 dither — same 0.1426 LSB realized RMS, same 2.033 % sparsity, same max of 1, same noise
band, only the direction randomised — damages **320 cells** where the trained refit damages **308**.

> **1,500 steps of gradient descent on the deployed grid bought 12 argmax cells over a coin flip of
> the same size — 0.51× the admit bar — while needing 430 cells in the other direction.** The random
> dither also improved **5** of 600 pairs; the trained refit improved **0**.

The axis is therefore closed by the GRID, not by the search. That is the difference between "we did
not search hard enough" and "the shipped format cannot express a seg-improving action at any
amplitude the carrier tolerates," and this arm is the first to separate them on the object that
ships.

**Pose is not the wall — and `ddm_ren1`'s own stated limit on that is now CLOSED in ren1's favour.**
ren1 measured the carrier's absorption on SYNTHETIC spectra and said so ("the perturbations are a
proxy for 'a refit moved the render by half an LSB,' not a refit"). Measured here on a REAL refit's
delta, n=120 seeded-random pairs against the control on the SAME pairs: the terminal carrier
re-solve takes the pose leg from **+3.85e-3 S (192 bars)** to **+1.01e-4 S (5.0 bars)** — a **38.2×**
removal — landing **1.0302×** the shipped level for **131 changed int12 coordinates**. ren1's
synthetic noise arm at 3.5× the amplitude lands at **1.0300×** on the same pairs. The absorption is
real and it transfers to a refit-shaped delta.

**So the full step-25 price decomposes, and SEG owns it:**

| term | ΔS | in admit bars | share |
|---|---:|---:|---:|
| **seg** (308 damaged cells) | **+2.6109e-4** | **13.1** | **64 %** |
| pose, post-re-solve | +1.0079e-4 | 5.04 | 25 % |
| member (+71 B) | +4.7276e-5 | 2.4 | 12 % |
| **net** | **+4.0917e-4** | **20.5** | — |

`ddm_pr1` and `ddm_rw1` were right about the axis and the campaign attributed the refusal to the
wrong term; ren1 corrected that and this arm measures it end to end. Pose is a quarter of the
refusal, not the whole of it and not nothing.

**Three corrections owed upward, all byte-identity receipts (§6):** the shipped semantic member's
Brotli container is **q=10 / lgwin=16** (not the `(ck2, 11, 24)` ren1 quotes from `ddm_fe1`; the
cited container encodes 91 B larger — **3.0× the admit bar** in score); the `--weight-qat-q3q4`
lever's low-bit set **disagreed with the shipped depth table** and has been made an input bound to
the packed bytes; and the current coded field differs from the DALI GT argmax at **18,900** sites,
not ren1's ft1-era 9,179.

---

## 1. DELIVERABLE 1 — the restored init, and the step-0 control

### 1.1 What was restored, read OUT OF THE PACKED BYTES

`rp1` r2 binds: *prove the realized quantizer from the packed bytes at step 0, not from the flags.*
Every row below was decoded from the move-48 archive by the shipped receiver
(`runtime/residual_archive` + `cpr1/ddm_mp2_semantic_receiver`), never inferred:

| property | MEASURED value |
|---|---|
| RX1M header | `('RX1M', 1, 2, 0, 250, 11629, 29862, 18450)` |
| container flags | `CK2_SEMANTIC_PLANE2` ON, `SZ1_SEMANTIC_SPLIT` off, `RC1_SEMANTIC_ADAPTIVE` on |
| SM3R header | version 1, **mode 6** (`MODE_ROW_PRUNE_MIXED`), **keep_percent 1**, reserved 0 |
| depth table (16 entries) | **3 bit** on `frame_embed.weight` and `blocks.0.film.weight`; **4 bit** on the other 14 — i.e. `sm3.PRUNE_MIXED_Q3_NAMES` |
| prune geometry | `blocks.{1,2,3}.film.weight` keep 2 of 192 rows: **[11, 13] / [34, 119] / [30, 189]** |
| scale rule | per-axis absmax / (2^(b−1)−1), stored fp16, floored at **5.960464477539063e-08** AFTER the cast; reduce axis `shape[-1]` for `*embed.weight`, else `shape[0]`; rank<2 tensors raw fp16 |
| SM1S mixer | 24 counted int8 weights `[29, 7, −25, −30, 9, 13, −8, −1, 7, 7, −7, 5, −4, 1, 1, 0, −1, 0, 4, −3, 13, 5, 11, 0]` |
| member container | **Brotli quality 10, lgwin 16**, identified by BYTE IDENTITY against the shipped stream (§6.1) |
| parameters | 66,339; nonzero 50,976 (76.8 %) |

Every one of these reproduces `ddm_ren1` §1.1 and §3A exactly. **ren1's provenance table is
independently confirmed, tensor by tensor, from the bytes.**

### 1.2 The init diff — what `ddm_ft1`'s warm start dropped

`ddm_ft1`'s `init_shipped_semantic_renderer.pt` (278,693 B, sha `460aa1c5…`) vs the restored init
(280,189 B, sha `ba51f59f3afd017baa2ac36a73493c8754d407fb3b734d8d712c609b4fd103b2`):

| | ft1 | restored |
|---|---|---|
| top-level keys | `{schema, state_dict, architecture_config, quant_bits, provenance}` | the same **+ `deployed_representation`** |
| the 38 weight tensors | present, and **verified identical to the deployed state** (0 mismatches) | identical |
| bit-depth table | **absent** — the scalar `quant_bits: 4`, a UNIFORM grid | the measured 16-entry table |
| `keep_percent` / prune geometry | **absent** | `keep_percent = 1`; the 190 zero rows per FiLM tensor are carried IN the state_dict itself and are the mask |
| scale rule + fp16 floor | **absent** | recorded |
| `provenance.archive_sha256` | `cbb8d928…` — **three moves stale** | `d830edd3…` (move 48) |

Exactly the three dropped objects MAIN's law predicts, confirmed on this object. One precision so
the fix is not overstated: the prune MASK is re-derived by the packer from the descending row norm,
so it is not a stored input — but the 190 exact zeros in the state_dict ARE the mask, and the
trainer's `--fixed-zero-mask` reads it from there (§2.2). The depth table and `keep_percent` are
literal arguments of `pack_prune_mixed_candidate` and were genuinely lost.

### 1.3 The step-0 CONTROL — it passes on all four legs

| leg | control | reference | agreement |
|---|---:|---:|---|
| SM3R body re-encode | 36,130 B `17e0fd0b…` | shipped | **byte-identical** |
| SM1S rider re-encode | 31,451 B `9c0b579c…` | shipped | **byte-identical** |
| RX1M member re-encode | 29,862 B `786950a5…` | shipped | **byte-identical** |
| realized state vs shipped | 0 tensors differ | — | exact |
| render, 120 seeded-random pairs | **0.0 LSB RMS**, 0 samples changed | shipped render | **bit-identical** |
| n600 `d_seg` through R | **0.00010338677300347222** | ren1 `0.00010338677300347` | **11 figures** |
| n600 `d_pose` | **4.586844968174052e-06** | ren1 resolve control `4.586845e-06` | **7 figures** |
| member Δ / archive Δ | 0 B / 0 B | — | exact |

Against the pointer's own T4 row the instrument sits 0.061 % low on seg and 0.069 % low on pose —
inside ren1's declared 0.06 %/0.07 % reproduction. **The control PASSED, so the arm proceeded.**

One honesty note the producer's own docstring got slightly wrong and this memo corrects: the pose
control reproduces ren1's **up2-instrument** value (`4.586845e-06`, the resolve producer's start) to
7 figures; ren1's step-0 PROBE measured `4.586758e-06` by a different path, 0.0019 % away. Both are
inside the tolerance; the sentence in the producer claiming it reproduces the probe specifically is
imprecise and the number above is the one to quote.

---

## 2. DELIVERABLE 2 — the trainer: what ran, what was fixed, what was refused

### 2.1 The fix that had to land first — F2's low-bit set was the WRONG set

`src/tac/pr130_lift/editability_levers.py` pinned `SELECTED_MIXED_Q3_NAMES =
{frame_embed, blocks.1.film, blocks.2.film, blocks.3.film}` — mirroring
`sm3.SELECTED_MIXED_Q3_NAMES`, the q3 set of the mp2 `MODE_ROW_PRUNE` candidate family. **The object
that ships is packed by `pack_prune_mixed_candidate` and its measured depth table is
`{frame_embed, blocks.0.film}` — `sm3.PRUNE_MIXED_Q3_NAMES`, a DIFFERENT set.**

So `--weight-qat-q3q4`, the lever ren1 named as the cure for ft1's trained-vs-realized divergence,
would have trained `blocks.0.film.weight` at 4 bits while the receiver realizes it at 3, and
`blocks.{1,2,3}.film.weight` at 3 bits while the receiver realizes them at 4 — the same defect,
relocated. This is the `rp1` r2 genus exactly: **a flag and a constant disagreeing silently.**

Fixed by making the set an INPUT: `EditabilityLeverConfig.weight_qat_q3_names` (default `None` ⇒ the
historical set, so every pre-existing config is byte-identical), sourced by the trainer from
`init["deployed_representation"]["bit_allocation"]` — the table read out of the archive's own bytes.
Six new tests, including one that pins the REASON (`PRUNE_MIXED_Q3_NAMES != SELECTED_MIXED_Q3_NAMES`)
so a future convergence of the two sets fails loudly instead of drifting. The run receipt now carries
the effective set, `q3_names_are_default: false`, and the provenance of the table.

Checked, not assumed, for the sister levers: F3's `FILM_ROW_FAMILY` and F1's `POSE_CRITICAL_TENSORS`
DO equal the shipped `PRUNE_NAMES`. Only F2's set was wrong.

### 2.2 `--film-row-dropout` is REFUSED here, with a measurement rather than an opinion

The charter names `--film-row-dropout --film-row-dropout-protect-top`. On THIS object that lever is
not dropout at all. MEASURED, and landed as a test:

* the deployed object prunes `blocks.{1,2,3}.film.weight` to **2 of 192** rows, so 190 rows are
  exactly zero and dropping them is a no-op;
* `film_row_order` selects exactly the two surviving rows, so `protect_top=2` keeps them on **every**
  draw;
* the inverted-dropout rescale therefore lands on them deterministically: every surviving nonzero
  entry is multiplied by **exactly 1/(1−p)** on every draw (measured at p = 0.1 / 0.5 / 0.9: gains
  1.111111 / 2.000000 / 10.000000, all entries equal, zero variance).

F3 would thus contribute **no stochastic content and a constant ×1/(1−p) gain the receiver does not
apply** — reintroducing the trained-vs-realized divergence it exists to remove. The charter's INTENT
(the trained object must be the realized object) is served by refusing it and using
**`--fixed-zero-mask`** instead, which is the built lever that makes the prune EXACT: measured after
training, the surviving rows are still `[11, 13] / [34, 119] / [30, 189]`, byte-identical geometry.

Declared over-constraint, so nobody discovers it later: `--fixed-zero-mask` freezes **every** exact
zero with `ndim ≥ 2`, which is 15,363 parameters — the 4,560 pruned-row entries plus **10,803**
coincidental zeros whose code happened to round to 0. That is 16.3 % of the parameters held fixed
that the receiver would happily let move. It is a constraint, declared, not a mechanism change.

### 2.3 What ran

Two trajectories, both from the restored init, both on the CODED field `a92e7d90…` (never GT-oracle),
both with the deployed mixed grid in the loop:

| | lr | steps | curriculum | batch | levers |
|---|---:|---:|---|---:|---|
| fine A | **2e-7** | 1,500 | pure `expected_flip` (`--ce-fraction 0 --softplus-fraction 0`), τ 0.15→0.05 | 2 | `--weight-qat-q3q4` (shipped table) + `--fixed-zero-mask` |
| fine B | **5e-7** | 1,500 | same | 2 | same |

plus a 3-lr bracket (2e-6 / 1e-5 / 5e-5 × 200 steps) used only to locate the amplitude ceiling.

Declared deltas from the reference form (ren1 §1.1's reconstruction of the shipped stage-08 trainer):
**SCOPE** — 1,500 steps instead of 6,000, and 2e-7/5e-7 instead of stage-08's 2e-7 only. The lr 2e-7
arm IS stage-08's lr and loss; the budget is a quarter of it. **NOT MECHANISM** — same loss family,
same quantizer (now actually the shipped one), same prune rule, canonical `tac.training.EMA` (decay
resolved from run geometry through `ema_decay_run_geometry_v1`), exact-R eval-roundtrip, per-stage +
periodic checkpoints every 25 steps with full optimizer/scheduler/RNG state, `--resume-from` present.
`steps_applied = 1500` in both receipts: the levers fired on every step, no orphaned lever.

Conditioning = the coded field; seg target = the **DALI** GT argmax, reused byte-identically from
ft1's cache after verifying it equals `jg1.load_gt_seg_labels(LINEAGE_DALI)` exactly. DALI because
that is the lineage the pricing instrument scores against; training and pricing must agree or the
run optimizes a different target than the one measured.

### 2.4 The in-loop signal: the surrogate INVERTS

The trainer's own advisory `quantized_exact_seg` (uniform-q4, not the realized grid — see §4.0):

| step | lr 2e-7 | lr 5e-7 |
|---:|---:|---:|
| 0 | 0.000113483 | 0.000113483 |
| 250 | 0.000114229 | 0.000116060 |
| 500 | 0.000114721 | 0.000118654 |
| 750 | 0.000115356 | 0.000120197 |
| 1000 | 0.000115712 | 0.000121290 |
| 1250 | 0.000115704 | 0.000121316 |
| 1500 | 0.000115670 | 0.000121596 |

**Monotone up on both trajectories: +1.93 % and +7.15 %. `best_step = 0` and
`improved_over_init = false` in both receipts — the argmin over 7 evaluations including step 0 never
left step 0.** The training LOSS falls over the same window (7.45e-4 → 4.70e-4 on lr 2e-7) but that
is the τ anneal deflating the surrogate, exactly the confound
[[tau_anneal_deflates_the_surrogate_read_losses_at_fixed_tau_20260904]] names; the loss is not
evidence and is not used as any.

---

## 3. DELIVERABLE 3 — the amplitude bound and the curve

The refit's realized render delta against the SHIPPED frame 1, in camera LSB RMS, split into
`ddm_ren1`'s bands (smooth = the 24×32 bicubic subspace the carrier lives on; noise = the residual),
on 12 seeded-random pairs per rung, 120 for every priced rung:

| run | step | LSB RMS | noise | smooth | noise/smooth | realized values changed | member Δ B | ≤ 0.5 LSB |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| lr 2e-7 | 25 | **0.1430** | 0.1406 | 0.0256 | **5.50** | 7,983 | +71 | ✔ |
| lr 2e-7 | 50 | 0.1744 | 0.1707 | 0.0352 | 4.85 | 11,956 | +8 | ✔ |
| lr 2e-7 | 75 | 0.2255 | 0.2172 | 0.0601 | 3.61 | 13,592 | +62 | ✔ |
| lr 2e-7 | 100 | 0.2565 | 0.2439 | 0.0789 | 3.09 | 14,624 | +27 | ✔ |
| lr 2e-7 | 150 | 0.2975 | 0.2784 | 0.1038 | 2.68 | 14,921 | +71 | ✔ |
| lr 2e-7 | 200 | 0.3296 | 0.3031 | 0.1280 | 2.37 | 15,511 | +95 | ✔ |
| lr 2e-7 | 250 | 0.3602 | 0.3248 | 0.1539 | 2.11 | 15,776 | +23 | ✔ |
| lr 2e-7 | 300 | 0.3885 | 0.3445 | 0.1775 | 1.94 | 16,087 | +70 | ✔ |
| lr 2e-7 | 375 | 0.4237 | 0.3656 | 0.2117 | 1.73 | 16,357 | +97 | ✔ |
| lr 2e-6 | 50 | 0.5234 | 0.4115 | 0.3192 | 1.29 | 19,209 | +63 | ✘ |
| lr 2e-6 | 100 | 0.6706 | 0.4493 | 0.4928 | 0.91 | 24,077 | +65 | ✘ |
| lr 2e-6 | 150 | 0.6565 | 0.4510 | 0.4715 | 0.96 | 26,475 | +12 | ✘ |
| lr 2e-6 | 200 | 0.6476 | 0.4505 | 0.4597 | 0.98 | 27,230 | +5 | ✘ |

Three things fall out, all MEASURED, all new:

1. **The delta is a sparse ±1 dither.** `max_abs_lsb = 1.0` at every admissible rung, and
   `fraction_changed = RMS²` to seven figures (step 25: 0.020364 vs 0.14270² = 0.0203642). The
   receiver's `clamp(0,255).round()` tail quantizes the refit's continuous weight move into a
   sparse binary one.
2. **A refit's own spectrum is the CHEAP end, and it degrades with amplitude.** noise/smooth falls
   monotonically from **5.50** at 0.143 LSB to **0.91** at 0.67 LSB. On this object smooth costs
   19.4× more pose than noise (ren1 §3.4, and obx2's law inverted here), so a small refit is
   pose-favourable and a large one progressively is not. Nobody had measured a REAL refit's spectrum
   before; ren1's two arms were synthetic fields at matched RMS.
3. **There is no zero-amplitude regime with a nonzero action.** 20 steps at lr 2e-7 change **zero**
   realized values and render bit-identically; 25 steps change 7,983 and land at 0.143 LSB. The
   ladder's first rung is already 7,983 values and 312 argmax cells (§4) — `ddm_rw1`'s grid-
   coarseness wall, seen from the amplitude side.

Bound enforced: the pricer REFUSES any checkpoint above 0.5 LSB unless `--price-outside-ceiling` is
passed, and no priced row used that flag. The lr 2e-6 and above rungs are recorded, not priced.

---

## 4. DELIVERABLE 4 — the price table, first rung first

### 4.0 What is priced, and on which object

Every priced row is the **REALIZED** state: the checkpoint's EMA shadow packed through
`pack_prune_mixed_candidate` → `SM1S` → `CK2` → Brotli(q10, lgwin16) and **parsed back by the shipped
receiver**, then rendered by the receiver's own batch-1 forward model and scored by the frozen CPU
SegNet through the evaluator's preprocess. The float checkpoint is never scored.

Stated so it is not mistaken for authority: the trainer's own `quantized_exact_seg` and its
`ema_deployed_argmax_parity` gate both run against **PR130's uniform-int4 `pack_semantic`** (which
reports a 40,252 B blob, not the deployed 36,130 B SM3R body). They are advisory in-loop signals
about a DIFFERENT packing, not the realized object; they are quoted in §2.4 as such and never used
to select or to price.

### 4.1 What a refit has to clear, re-derived at move 48

`∂S/∂d_seg = 100`; `∂S/∂d_pose = 5/√(10·4.59e-6) = 738.02`; `1 B = 6.658589531221714e-7 S`;
`1 cell = 100/117,964,800 S`, so the −2e-5 bar is **23.59 repaired cells** and the pointer's own
error is **12,203.5 wrong cells**. ren1 §4.1b's required cuts re-derive here to 3 figures:
noise-spectrum residue `+8.3963e-5 S` (99.0 cells) + bar ⇒ **1.005 %**; smooth `+2.5727e-4 S`
(303.5 cells) + bar ⇒ **2.683 %**. Member fees seen on this ladder (+5 … +97 B) are **3.9 … 76.2
cells** — small, and noisy at the scale of the 34.8 B container lottery.

### 4.2 The measured rows

| checkpoint | LSB RMS | band | member Δ B | n600 `d_seg` | vs control | net cells | gross cells moved | pairs improved / worse / same |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **control** (restored init) | 0.0000 | — | 0 | 0.00010338677300 | — | 0 | 0 | — |
| lr 2e-7 step 25 | 0.1427 | noise 5.5:1 | +71 | **0.00010599772135** | **+2.525 %** | **−308.0** | 312 | **0 / 242 / 358** |
| lr 2e-7 step 100 | 0.2562 | noise 3.1:1 | +27 | **0.00010776943631** | **+4.239 %** | **−517.0** | 540 | **2 / 333 / 265** |
| lr 2e-7 step 375 | 0.4235 | noise 1.7:1 | +97 | **0.00011059231228** | **+6.969 %** | **−850.0** | 911 | **6 / 450 / 144** |
| **random ±1 dither** @ 0.1426 | 0.1426 | noise | — | **0.00010609944661** | **+2.624 %** | **−320.0** | — | **5 / 251 / 344** |

**The refusal is not marginal.** The required cut is **−1.005 %**; the measured cut is **+2.525 %** —
a gap of **3.53 percentage points**, i.e. the refit must find **430 net repaired cells** and instead
**damages 308**. The seg leg is measured before the expensive pose leg precisely so a candidate its
own cheapest binding term has already refused does not consume a terminal re-solve; the step-25
candidate was nonetheless taken to a re-solve BRACKET (§4.4), because that measurement answers a
question `ddm_ren1` left open about its own result and is worth having even on a refused row.

Margins in the two units the charter asks for:
* in **34.8 B container-lottery** units: the step-25 row's seg damage alone is
  `308 cells × (100/117,964,800) = +2.611e-4 S`, which is **13.1× the admit bar** and **11.3
  lotteries** (1 lottery = 34.8 B = 2.317e-5 S). The member fee (+71 B = 4.73e-5 S) is **2.0
  lotteries**. The seg term dominates the byte term **5.5:1**; this is not a byte-noise verdict.
* in the **instrument's 0.06 % seg reproduction**: the measured +2.525 % is **41.6×** the
  reproduction gap. Resolvable by a wide margin.

### 4.3 The per-pair ORACLE bound — a door priced shut before anyone builds it

A weight change has no per-pair admission lever (ren1 §4.2): it moves all 600 frames. A successor
could ship a per-pair SELECTOR — 600 bits = 75 B, a receiver change and therefore not this arm's to
fire. Priced from the per-pair rows so nobody spends a receiver change to find out:

| checkpoint | oracle `d_seg` | oracle cut | ΔS incl. 75 B selector | clears the bar |
|---|---:|---:|---:|:--:|
| step 25 | 0.00010338677300 | **0.0000 %** | **+4.99e-5** | ✘ |
| step 100 | 0.00010336981879 | 0.0164 % | +4.82e-5 | ✘ |
| step 375 | 0.00010333591037 | 0.0492 % | +4.49e-5 | ✘ |

At step 25 the oracle IS the shipped renderer, because **no pair improves**. At the larger amplitudes
a handful of pairs improve by 1–2 cells, and the oracle buys 2 and 6 cells respectively — against a
75 B selector costing **58.9 cells**. **The per-pair-selector door is priced shut on this candidate
family**, by a factor of 10–30, and nobody needs to spend a receiver change to learn it.

---

## 4.4 The pose leg, MEASURED on a real refit — and `ddm_ren1`'s own limit CLOSED in its favour

`ddm_ren1` stated its own honest limit: *"The perturbations are a proxy for 'a refit moved the render
by half an LSB,' not a refit."* This arm measures the real thing on the step-25 candidate.

### 4.4.1 The measurement

`up2.solve_pair_realized`, the canonical unforked uncapped greedy descent on the REALIZED objective,
driven through a frame-1 shim that renders the CANDIDATE — the same shim shape ren1 used. Seeded
RANDOM n=120 via `up2.select_pairs` (never a prefix; the pose axis is where prefix bias is
anti-conservative at 2.54–4.21×). Every comparison is against `ddm_ren1`'s retained per-pair control
rows **restricted to the same 120 pairs**, so no cross-pair-set subtraction is involved.

| arm | amplitude | start `d_pose` | final `d_pose` | final / control | recovery | changed int12 coords |
|---|---:|---:|---:|---:|---:|---:|
| control (shipped) | 0 | 4.526114e-06 | 4.522750e-06 | 1.0000 | 1.00× | 0 |
| **refit, step 25** | **0.1427 LSB** | **1.118290e-05** | **4.659338e-06** | **1.0302** | **2.40×** | **131** |
| ren1 noise (synthetic) | 0.5 LSB | 3.264278e-05 | 4.658239e-06 | 1.0300 | 7.01× | 159 |
| ren1 smooth (synthetic) | 0.5 LSB | 7.128214e-04 | 4.883812e-06 | 1.0798 | 145.96× | 247 |

`all_converged: true` — every pair exited on "no improving lattice neighbour exists", the solver's
own convergence proof, not on a pass cap. 94 of 120 pairs improved.

### 4.4.2 What it says, and what it does not

* **ren1's absorption CAPACITY transfers to a real refit.** A refit-shaped delta at 0.1427 LSB lands
  at 1.0302× shipped — indistinguishable from ren1's synthetic noise arm at 1.0300×, measured on the
  same pairs. ren1's honest limit #2 is closed, in ren1's favour.
* **The residue is a LATTICE floor, not a function of amplitude in this band.** The refit's
  amplitude is 3.5× smaller than ren1's noise arm and its post-re-solve residue is the same 3 %. That
  is what "capacity, not ratio" predicts: inside capacity you land on the int12 lattice's own floor,
  and the floor does not shrink with the perturbation.
* **Pose therefore costs a real 5.0 bars here, not 0 and not 170.** Quoting ren1's "+2.5 % of
  shipped for 2–4 B" as "free" would be the stale-headline genus; 3 % of shipped IS +1.0e-4 S at this
  operating point because `∂S/∂d_pose = 738`.
* **Not measured, and stated as such:** the carrier's own byte delta. This producer prices it with
  `up2.price_full_resolve_bytes` only at n600 (a partial code set would be a fake price, so it
  returns `null` for a bracket). Bounded by ren1's n600 anchor — 824 changed coordinates cost **+2 B**
  — and this bracket's 131 coords over 120 pairs extrapolate to ~655 over 600, so **≤ +2 B ≈ 0.1
  bar**. It cannot rescue a 20.5-bar refusal and it is not counted in the table above.
* **Scope:** n=120 seeded random is a BRACKET, not an n600 verdict. It is control-relative on the
  same pairs, which removes the pair-hardness confound entirely, but a successor quoting the 1.0302×
  should re-measure at n600 before making it load-bearing.

---

## 5. THE FALSIFIER — is the trained direction better than a coin flip of the same size?

This is the control that decides whether the negative is about the SEARCH or about the GRID.
`ddm_rw1` ran a random-direction control in its own formulation; this is that control lifted onto
THIS one and matched to the refit's realized statistics.

**The match is on the REALIZED delta, and getting that wrong was a measured mistake worth
recording.** The first construction scaled ren1's iid Gaussian to 0.1430 LSB and handed it to the
receiver's `clamp(0,255).round()` tail. The camera raster is integer, so a σ = 0.143 field leaves a
sample unchanged unless `|x| > 0.5`: the realized delta came back at **0.0216 LSB RMS over 0.047 %
of samples** instead of 0.1430 over 2.04 % — a **6.6× under-shoot** that would have made the control
look harmless for a reason with nothing to do with direction. Caught by reading the realized numbers,
not the target. The corrected control is a sparse **±1 dither** at density RMS², which reproduces the
refit's realized delta by construction: measured **0.14260 LSB RMS / 2.0335 % changed** against the
refit's **0.14270 / 2.0364 %** — matched to 0.07 % and 0.14 %.

### 5.1 The result, n600, on the same 600 pairs

| | `d_seg` | vs control | cells damaged | pairs improved / worse / same |
|---|---:|---:|---:|---|
| control | 0.00010338677300 | — | 0 | — |
| **trained refit** @ 0.1427 LSB | 0.00010599772135 | **+2.5254 %** | **308** | **0 / 242 / 358** |
| **random ±1 dither** @ 0.1426 LSB | 0.00010609944661 | **+2.6238 %** | **320** | **5 / 251 / 344** |

> **refit damage / random damage = 0.96250.** Fifteen hundred steps of gradient descent on the
> deployed grid bought **12 argmax cells** over a coin flip of the same size — **0.51× the admit
> bar** — in a problem that needs **430 cells in the other direction**.

Two details that sharpen it rather than soften it:

* the random dither **improved 5** of 600 pairs; the trained refit improved **0**. The search did not
  merely fail to find enough direction — on the per-pair improvement count it did worse than chance.
* the damage scales as **amplitude^0.9** across the three priced rungs (308 → 517 → 850 cells at
  0.1427 → 0.2562 → 0.4235 LSB), i.e. nearly linear in RMS and sub-linear in the changed fraction.
  There is no amplitude at which the damage curve bends toward zero faster than the required cut.

### 5.2 What this separates

`ddm_rw1` closed at FORMULATION scope on grid resolution: *"the fold-back is limited by the grid's
resolution, not by the search."* It said so from a coordinate search without the depth table or the
prune mask in the loop, on older fields. This is the same conclusion reached by a completely
different route — a gradient trajectory, with the deployed grid and prune actually realized, on the
current field, against a matched random direction — and it is now a statement about the FORMAT rather
than about anybody's search budget:

> **The deployed SM3R mode-6 grid cannot express a seg-improving action for this renderer on this
> field.** Its smallest realizable action moves ~312 argmax cells and lands them essentially at
> random; 98.7 % of them land wrong, which is what "random" looks like when the incumbent is already
> a conditional optimum.

---

## 6. CORRECTIONS OWED UPWARD

### 6.1 The shipped semantic member's Brotli container is **q=10 / lgwin=16**, not `(ck2, 11, 24)`

ren1 §4.4 quotes `ddm_fe1`: *"the shipped shape pinned by BYTE identity at `(ck2, 11, 24)`."* On the
move-48 archive that is not the container. Searched `q ∈ {9,10,11} × lgwin ∈ {16,18,20,22,24}` and
compared BYTES, not sizes:

| container | bytes | Δ vs shipped | byte-identical |
|---|---:|---:|:--:|
| **ck2, q10, lgwin16** | **29,862** | **0** | **✔** |
| ck2, q11, lgwin24 | 29,953 | **+91** | ✘ |
| ck2, q11, lgwin16 | 29,953 | +91 | ✘ |
| ck2, q11, lgwin22 | 29,953 | +91 | ✘ |
| ck2, q9, lgwin16 | 30,489 | +627 | ✘ |

A successor pricing a semantic member with the cited container carries a **+91 B systematic offset =
6.06e-5 S = 3.03× the admit bar**, in the wrong direction. Note also that q11 is WORSE than q10 on
this payload, so "max quality" is not the shipped choice and must not be assumed. This is the
binding-numbers-expire genus: fe1's pin was true of the object fe1 measured.

### 6.2 `--weight-qat-q3q4` trained the wrong tensors at 3 bits

§2.1. Fixed at the module, bound to the packed bytes, six tests, default preserved.

### 6.3 The coded field differs from the DALI GT argmax at **18,900** sites, not 9,179

ren1 §1.1 carries *"The shipped field differs from DALI GT at 9,179 sites."* MEASURED on the current
field `a92e7d90…` against `jg1.load_gt_seg_labels(DALI)`: **18,900 of 117,964,800 (1.6022e-4)**.
ren1's 9,179 is cited to `ddm_ft1` (2026-09-03) and belongs to the pre-move-31 field. The direction
of ren1's reading is unchanged and in fact strengthened: the field disagrees with GT **1.55× more**
than the rendered output does (1.6022e-4 vs `d_seg` 1.0339e-4), which is what an optimized pre-image
looks like.

### 6.4 A smaller one, about this arm's own producer

`ddm_ren2_price_checkpoint.py`'s pose-leg note claims it reproduces ren1's **step-0 probe** control;
it reproduces ren1's **up2-instrument** control (the resolve producer's start) to 7 figures and the
step-0 probe to 0.0019 %. §1.3 carries the accurate statement. Also recorded rather than silently
patched: `ddm_ren2_restore_init.py` writes the restored init and the 118 MB conditioning cache
through `torch.save` rather than its own reserve-checked `retain()`, so those two writes are not
reserve-gated (every other write in the arm is, and the arm-level `RETENTION.json` checks the
reserve). The code was left exactly as it ran rather than edited after the fact.

---

## 7. VERDICT

**The pointer did not move and this arm shipped no candidate.** No checkpoint cleared the bar; no
seal inputs were staged; no byte-close was built, because deliverable 5 is conditional on a
checkpoint clearing and none did.

**What is now closed, and at what scope.** The refit-in-place of the deployed renderer, on the
current coded field, with the RESTORED init, the DEPLOYED mixed grid actually in the loop, the prune
exactly preserved, canonical EMA, exact-R roundtrip, priced through the shipped receiver at n600:
**REFUSED**, at **FORMULATION** scope — one loss family (`expected_flip` tail), two learning rates,
one budget (1,500 steps), one optimizer, one object. It is NOT a family kill. What makes this
different from `ddm_ft1` (which refit a different object) and `ddm_rw1` (which searched without the
depth table or the prune mask, on older fields) is that the object under test here IS the object
that ships, proved byte-for-byte at step 0.

**Where the wall actually is.** Not pose — the terminal re-solve removes 38.2× of the pose leg and
leaves 5.0 bars, a quarter of the refusal. Not bytes — the member fee is 3.9–76.2 cells against a
430-cell demand, and the carrier's is ≤ 2 B. **The wall is that the deployed grid's smallest
expressible action already moves ~312 argmax cells and the gradient cannot aim them**: 98.7 % land
wrong, 0 of 600 pairs improve, and an amplitude-matched random dither does the same thing 3.75 %
worse. The refusal is a property of the FORMAT's resolution, not of the search budget.

### What the next unit should run, specified

1. **Do not re-run a seg-objective refit of this renderer at these amplitudes.** Two trajectories,
   nine admissible rungs, three n600 priced rows, 0/600 pairs improved. A third lr is not new
   information; a different MECHANISM would be.
2. **The live question this leaves is the JOINT one, and it is now cheap to ask.** The field was
   optimized against this θ eight times (moves 24–48 are token edits); θ has never moved. The refit
   fails because it is a conditional optimum of a joint problem being perturbed on one coordinate
   only. The successor is **alternating**: refit θ, then RE-SOLVE the token field against the new θ
   with the existing token machinery, and price the pair. Every piece exists; the composition does
   not. That is a real door and this arm did not open it.
3. **`ddm_ntb2`'s named prize is still open and is now correctly priced.** Recovering the
   `+4.92e-05 S` that the 3-bit `blocks.0.film` choice cost is 0.48 % — under the 1.005 % a
   noise-spectrum refit must buy, so it clears only if composed or if the render change is smaller
   than 0.143 LSB, and §3 measures that there IS no smaller nonzero rung on this grid. Per-row mixed
   depth (a receiver change) is the mechanism that would create one.
4. **Reuse, do not rebuild:** the restored init, the byte-exact member pricer, the amplitude curve
   producer, the matched random-direction control, and the per-pair oracle report are all landed and
   all bound by sha. The member pricer in particular is the first one in this campaign proved
   byte-identical to the shipped encoder.

---

## 8. BOUNDARIES AND CUSTODY

Code landed at commit `0215c504d`: `experiments/ddm_ren2_{restore_init, price_checkpoint,
amplitude_curve, seg_delta_report, random_direction_control, retention}.py`,
`src/tac/pr130_lift/{editability_levers, train_semantic_quantized_resumable}.py`,
`src/tac/tests/test_editability_levers.py` (49 tests, all passing). HEAD at arm start
`22173671f`. Trainer file sha at arm start `8ed7105e7e8ce82ad27f21db8c66a2becdc7a71d92197e15f333af8737f6aae1`.
Lane `ddm_ren2_renderer_refit_in_place_coded_field_20260912`, claimed.

One provenance note, recorded rather than hidden: `ddm_ren2_price_checkpoint.py` changed twice
mid-arm, both times to close a FAIL-OPEN guard rather than to change a number — (a) `fires` now
additionally requires an n600 resolve leg and a MEASURED carrier byte delta, so a subsample bracket
cannot set it; (b) a resumed row set is now refused when the label already holds rows for a different
checkpoint sha. Neither edit can alter any row already produced (every affected predicate evaluates
to the same value on a seg-only leg), and each run's `INPUTS.json` records the producer sha that
actually ran.

- **No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`, no paid dispatch, no contest
  evaluation, no seal call, no packet, no candidate archive.** MAIN fires everything.
- Nothing edited under `upstream/`, `submissions/semantic_joint_ctxmix/`, any sealed tree,
  `src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, or any receiver code. The member's
  VALUES are the only thing this arm ever changed, and no candidate member was shipped.
- `ddm_ren1`'s and `ddm_dpi1`'s directories were **read only** (ren1's retained `planes.npz` supplies
  the shipped argmax plane for the cells-moved column; nothing was written there). `ddm_ft1`'s
  retained init and DALI target cache were read, never written.
- **No storage reserve lowered.** Every producer keeps the 40 GiB fail-closed reserve.
- **Every payload on the SSD tier** under `/Volumes/VertigoDataTier/pact/ddm_ren2/`, sha-verified in
  `RETENTION.json`: **239 payloads, 374,210,782 B = 0.3485 GiB** against the 8 GiB cap, 52.5 GiB free
  after, reserve respected. That includes all 120 trainer checkpoints (both trajectories, every 25
  steps, full optimizer/scheduler/RNG state), every per-pair `seg_rows` / `pose_rows` /
  `resolve_rows` JSONL, every `INPUTS.json` / `RESULT.json`, the restored init and the conditioning
  cache. Local disk held source only. The ~1.8 GB camera rasters are NOT retained: they are exactly
  rebuildable from the pinned archive plus the pinned token field by producers whose shas are
  recorded in each `INPUTS.json`, and the tier holds a 40 GiB reserve every producer refuses to cross.
- Every heavy launch went through `tools/launch_detached_process.py` with an armed done-receipt, and
  every trainer additionally through `tools/safe_run.py` (admission is ENFORCED on this box; a raw
  launch is refused). Heavy waits were receipt-bound until-loops, never a foreground clock.
- MPS was the training-gradient device only; every score-relevant number is CPU, and no number here
  is a score.

The frontier is unchanged: **composition S 0.13638261682704697 @ 179,111 B `[contest-CUDA T4 n600]`
(move 48)**.
