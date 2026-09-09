# ddm_rw1 — the joint renderer fold-back on the shipped object: the actuator is the int4 CODE, the rate corner is open at 0.014 B/code, and the incumbent falsified my own closed form before any training step (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm `ddm_rw1` · Lane `lane_ddm_rw1_renderer_edge_foldback_20260909` · Charter
`.omx/research/charters/ddm_rw1_boundary_local_renderer_weight_foldback_per_pair_admitted_20260909.md` · Gestalt: gs3 Addendum 15.

Axes: d_seg `[macOS-CPU advisory, cpu_torch argmax, DALI lineage, n600]` · d_pose `[cpu_torch fp32 authority, DALI GT, n600]` ·
training `[macOS-MPS research-signal]` · bytes EXACT (real container-searched encode) · `score_claim=false` throughout; no T4 row exists
for this arm and MAIN fires.

---

## 0. Headline

**The pointer did NOT move on this arm.** The reach IS measured, it is negative, and the arm closes at `verdict_scope: formulation` with
its cause named and quantified. Stated at the operating point as every verdict here must be (S 0.13882326433317044, gap to 0.12 =
0.018823264):

| | ΔS | share of gap |
|---|---|---|
| the residual this arm targeted (12,866 cells) | 0.010907 | **57.9 %** |
| rate toll for rewriting all 12,672 codes (+176.7 B) | 1.177e-04 | **0.63 %** |
| best trained delta, 66 codes (reach **−11.37 %**) | **+1.240e-03** | **+6.59 %** |
| discrete code search, 150 evals | **0.0** | **0.00 %** |
| fp16 scale search (door 1), 91+ evals | **0.0** | **0.00 %** |

**The closing law** (`renderer_edge_layer_foldback_reach_v1`): *one int4 code step in `head`/`blocks.3` already moves 240–455 argmax
cells, so the fold-back is limited by the grid's resolution, not by the search.* Door (1) then measured a **41× finer** actuator with a
**negative** rate break-even and still found nothing, and the damage exponent it fitted (**0.073–0.568**, roughly a square root) closes
door (2) by arithmetic. §8b–§8d carry the four routes; §10 the two registered laws.

Three findings survive the closure and travel:

- **The rate corner for a renderer-weight fold-back is essentially open.** Rewriting the shipped signed-int4 codes of
  `head.weight` + `blocks.3.dw.weight` + `blocks.3.pw.weight` costs **0.01398 B per changed code** — `+176.7 B` to rewrite ALL
  **12,672** codes, and only `+58.3 B` at N = 6,000, with N = 50 and N = 200 measuring **negative**. That is **10.7× cheaper per code**
  than fe1's `model_section_edit_container_break_fee_v1` (0.15 B/code, fitted on the depth-3 `frame_embed` run at N ≤ 200), and it drops
  the break-even from 0.1178 to **0.011 repaired seg cells per changed code**. A FULL rewrite of the three tensors is paid by
  **139 repaired cells — 1.08 % of the 12,866-cell residual**. The charter's falsifier (a) threshold (reach ≥ 3 % ⇒ 386 cells) therefore
  pays the rate **2.8× over**.
- **The pose re-solve has full structural headroom on every pair.** All 600 per-pair Jacobians `∂pose/∂coeff` are **rank 6 of 6**
  (12 free coefficients against 6 scored dimensions), and the distance from the live codes to the int12 lattice edge is
  **1,892–2,020 code units**. There is no rank-deficient pair and no pair near the lattice edge.
- **My own closed-form pose floor was falsified by the incumbent, immediately.** See §4. It is relabelled, not defended.

What the arm exists to buy — the **reach** — was measured on the live body after the gate opened. It is negative on every route tried:
a 3,000-step joint run (§8b), a random-direction control (§8b), the exact-field gradient in both ranking directions (§8c), a discrete
realized code search (§8c), and a per-row fp16 scale search (§8d). Five routes, one cause.

*(§0 was rewritten when the reach landed; the pre-gate framing it replaced said "NOT YET MEASURED", which was true when written and
would have been a stale headline over a corrected body if left — [[m106]].)*

---

## 1. What the actuator IS, and why it is not what ft1 trained

ft1 measured the realization gap directly rather than assuming it away: its trainer applied a **uniform int4 fake-quant with no row
prune**, while the deployed SM3R section uses per-tensor depths `{3, 4}` and keeps **2 of 192 rows** in each of
`blocks.{1,2,3}.film.weight`. The trained object and the realized object differed by `2.32e-3` max-abs
(`retained/verdict_ft1_step600.json` → `export.trained_vs_realized_max_abs_delta`).

This arm removes that gap **by construction rather than by measurement**: the trainable parameter IS the shipped signed-int4 code, the
shipped fp16 per-row scales are FROZEN, and only runs that are (a) depth 4 and (b) NOT row-pruned are opened. On the live section that
is exactly (measured, `receipts/CONTROL.json`):

| tensor | shape | codes | code bytes | scale bytes | depth | row-pruned |
|---|---|---:|---:|---:|---:|---|
| `head.weight` | (3, 96, 3, 3) | 2,592 | 1,296 | 6 | 4 | no |
| `blocks.3.dw.weight` | (96, 1, 3, 3) | 864 | 432 | 192 | 4 | no |
| `blocks.3.pw.weight` | (96, 96, 1, 1) | 9,216 | 4,608 | 192 | 4 | no |
| **total** | | **12,672** | **6,336** | **390** | | |

`blocks.3.film.weight` is in `ROW_PRUNE_NAMES` and ships **2 of 192 rows** at `keep_percent = 1`; the module REFUSES to edit a pruned run
for exactly the ft1 reason, so the FiLM half of the last block is out of scope by construction and is declared as such, not silently
skipped.

Consequence: **"trained == realized" is an identity here, not a measurement.** There is no re-quantization step between the trainer's
object and the shipped bytes.

---

## 2. The three identity controls (all PASS, all on the rc2 base)

| control | what it proves | result |
|---|---|---|
| **render identity** (`receipts/CONTROL.json`) | the lifted forward IS the receiver's forward | `max_abs_delta = 0` over **24,416,064** pixels, 8 spread pairs (0, 1, 2, 137, 299, 300, 450, 599), `semantic_batch = 1` |
| **packer identity** (`receipts/SECTION.json`) | repacking the SHIPPED codes reproduces the shipped bytes | SM3R body **36,130 B** byte-identical (sha `17e0fd0b…`), RC1 stream **31,792 B** byte-identical, `changed_codes_on_identity = 0` |
| **null build** (`receipts/EXPORT_NULL.json`) | the whole archive path is an identity | **181,414 B**, sha `c810c2c7…`, `bytes_vs_live = 0`, container `("ck2", 11, 24)`, `rx1_reserved = 0x7a` |

Without the third one no byte number below would be attributable to the weight change rather than to the repack. `semantic_batch = 1`
is not a performance oversight: up2 §6 MEASURED batch 8 as byte-changing on this half (1,326 pixels by ±1).

### 2.1 The re-base to the 33rd pointer move, measured rather than inherited

The pointer moved to rc2 mid-arm (S `0.13885056455024844` @ 181,414 B, sha `c810c2c7…`). MAIN reported that only the hpac model section
changed. That is a claim about bytes, so it is CHECKED here (`verify_shared_decode_sections`, run on every stage):

| section | pass-3 tree | rc2 tree | verdict |
|---|---:|---:|---|
| `semantic_blob` | 31,792 | 31,792 | **identical** |
| `carrier_blob` | 18,931 | 18,931 | **identical** |
| `token_stream` | 120,225 | 120,225 | **identical** |
| `residual_payload` | 100 | 100 | **identical** |
| `hpac_blob` | 16,267 | 16,061 | differs (−206) |

Because the four decode-determining sections are byte-identical AND rc2's T4 row reproduced `d_seg 0.00010913` / `d_pose 5.1e-6`
exactly, the pass-3 parse-back decode is reused as the live decode instead of re-inflating, and the pinned cell count (12,866) and pose
(5.0928018072772644e-06) carry across the move unchanged. Both identity controls were re-run on the rc2 base after the move and both
re-passed.

---

## 3. THE RATE LAW, re-derived at this scope (the measurement that changes the arm)

`receipts/RATE_LAW.json`, 3 repeats per point, seeded (20260909), priced by a **real container-searched encode** of the semantic section
(30 shapes: `{ck2, plain} × q ∈ {9,10,11} × lgwin ∈ {16,18,20,22,24}`, ties to the shipped shape). Every other section is byte-identical
under this arm's change, so **the archive delta IS the semantic-stream delta, exactly**.

| N codes changed | mean Δ bytes | B / code | min | max | cells to break even |
|---:|---:|---:|---:|---:|---:|
| 1 | +17.0 | 17.00 | −9 | +50 | 13.4 |
| 10 | +16.0 | 1.600 | −3 | +51 | 12.6 |
| 50 | **−21.3** | −0.427 | −36 | −4 | −16.8 |
| 200 | **−30.7** | −0.153 | −35 | −25 | −24.1 |
| 1,000 | +0.3 | 0.0003 | −25 | +17 | 0.3 |
| 3,000 | +57.3 | 0.0191 | +27 | +78 | 45.0 |
| 6,000 | +58.3 | 0.0097 | +52 | +62 | 45.8 |
| **12,672 (all)** | **+176.7** | **0.0139** | +119 | +212 | **138.8** |

Equations leg (`tac.canonical_equations`): this is registered as a third empirical anchor on
**`model_section_edit_container_break_fee_v1`** — anchor
`rw1_semantic_int4_weight_code_rewrite_marginal_0p0139_bytes_per_code_20260909`. It is a **domain
extension, not a contradiction**: that law's own `excluded` list names *"extrapolation past N = 200
changed codes"*, and this anchor closes exactly that gap on a different run (depth-4 weight codes
rather than depth-3 `frame_embed`) at N = 1…12,672. The fee **saturates**.

Linear fit over the eight points: **0.01398 B/code**, intercept −5.87 B (the intercept is an artefact of a fit dominated by large N — a
zero-code change costs zero by construction; read the table, not the intercept).

**Reading.** fe1's law (`0.15·N + 8` B searched) was fitted on the **depth-3 `frame_embed`** run at **N ≤ 200**. Extrapolating it to
N = 12,672 would have predicted ≈ **1,909 B**; the measured cost is **176.7 B**, i.e. the extrapolation over-charges **10.8×**. This is
the cross-regime-constant-transfer genus ([[m143]]) caught by re-deriving rather than by carrying: the fee is a *container-break* effect
that saturates, not a per-code price, and int4 code runs at this scale are close to incompressible in both directions, so re-writing them
neither helps nor hurts brotli much.

**Consequence for the arm.** One repaired seg cell buys `8.477105034722222e-07` S; one archive byte costs `6.658589531221714e-07` S.
Break-even is therefore **0.011 repaired cells per changed code** — the arm may rewrite EVERY code it owns and still only owes 139 cells.
The charter's framing ("head 1,296 B raw … ≈ 12 KB raw at int4 … price BOTH forms, replace vs coded delta") is superseded by measurement:
the **replace form costs 177 B for a total rewrite**, so a coded sparse/low-rank delta is unnecessary and would be strictly worse
(it would add a receiver change for a saving of at most 177 B). ITEM "both rate forms" is CLOSED by measurement, in favour of replace.

---

## 4. The pose geometry — and my own closed form, falsified by the incumbent

`receipts/PREP.json` (n600, cpu_torch fp32, DALI GT, 637 s). Per pair: the 6×12 Jacobian `∂pose/∂coeff`, the min-image-norm re-solve
operator `A = G⁻¹Jᵀ(JG⁻¹Jᵀ)⁻¹ / scales` in CODE units, the base pose, and the distance to the int12 lattice edge.

| quantity | measured |
|---|---|
| d_pose recomputed on the live decode | `5.092826353616941e-06` |
| live receipt | `5.0928018072772644e-06` |
| relative error | **4.8e-06** (the pose instrument reproduces the live row) |
| rank-deficient pairs (rank < 6) | **0 of 600** |
| reach budget to the int12 edge | min **1,892**, median **1,972**, max **2,020** code units |

**Structural finding:** twelve free carrier coefficients against six scored pose dimensions is generically surjective, and it is
surjective on **every one of the 600 pairs**. So at first order the per-pair re-solve can cancel ANY pose residual a render change
produces, and it can do so without approaching the lattice edge. That is the mechanism behind fe1's measured 643–3,053× per-pair
recovery, stated structurally instead of anecdotally.

### 4.1 The falsification

I derived a closed-form "lattice floor": after a perfect re-solve the realized codes are rounded, leaving `δ ~ U(−½, ½)·scale` per
coefficient, so `E‖Jδ‖²/6 = ((J·scale)²).sum() / 72`. Measured mean: **1.288e-4**, i.e. **25.3× the live d_pose**, predicting a
post-re-solve pose leg of `√(10·1.288e-4) = 0.0359 S` against the live `0.00714 S`.

**The incumbent falsifies this the moment it is written.** The live row already sits at `d_pose = 5.0928e-06` **on that same lattice**,
25.3× BELOW my bound. So the bound is not the achievable floor — jg5's `refine_pair` does not merely round; it alternates Gauss-Newton
with a **±2 integer polish**, and that polish beats uniform rounding by at least that ratio. The quantity is relabelled in the receipt as
`naive_rounding_pose_bound` with `incumbent_falsifies_naive_bound: true` and its looseness recorded; it is **withdrawn as a prediction**.
The operative expectation for the post-re-solve leg is fe1's MEASURED per-pair recovery band (resolved / base ∈ 0.24–1.36×), and that band
is what stage D will test on a global change.

This is the zero-gravitational-pull rule doing its job on my own derivation: a closed form that the incumbent already beats is not a
floor, it is a loose bound, and saying so costs nothing while defending it would have cost the pose leg.

### 4.2 What the falsification did to the loss

The pose barrier I derived from the reach budget is consequently **near-vacuous**: measured at **1e-12 to 1e-14** across the smoke, because
the demanded carrier step is ~0.0001–0.007 code units against a budget of ~1,900. A barrier that never binds is not a pose term.

So the loop carries **two** pose terms, both derived:

1. **the barrier** — weight `100 · d_seg_live = 0.0109`: at full reach the correcting step leaves the lattice and the pair becomes
   unpayable, which is worth exactly the whole seg residual the arm is trying to buy. Retained because it is the physical constraint,
   and its measured vacuity is a finding, not a defect.
2. **the stale term** — weight `(5/√(10·d_pose_live)) / 643 = 1.0896` per unit of stale d_pose: the S-linearisation of the pose leg at
   the live operating point, DISCOUNTED by fe1's **worst** measured per-pair recovery (643×, pair 382). The conservative end of measured
   evidence, never the best case and never a guess.

---

## 5. The loss, derived from the score rather than from the census

The seg term is ft1's terminal form, `sigmoid(−margin/τ)` with `margin = gt_logit − max(other)`, through the exact R
(bilinear → 874×1164 → clamp/round with a straight-through gradient → SegNet's own `preprocess_input`) against the **DALI** table.

Two deliberate departures from the charter's instruction, both closed-form:

- **No band mask.** The census says rows 128–319 carry 100.00 % of the residual — but `sigmoid(−margin/τ)` at τ ≤ 0.15 is already ~0 for a
  confident cell, so the sigmoid IS the at-risk selector and a hand-applied band would only remove the loss's ability to *see* damage it
  creates outside the band. The band is kept as a diagnostic.
- **No class up-weighting.** The census's Lane 40.15× / Movable 10.92× enrichment is real, but **one flipped cell costs the same
  `8.477e-07` S whatever its class**. A per-class multiplier would optimise a different objective than S. Deriving the weight from the
  score rather than from the census is the closed-form-first answer, and the enrichment stays a diagnostic. (The knob exists and is off.)

### 5.1 The learning rate, derived from the actuator and then MEASURED

hr1's standing instruction is a *current-vehicle update-norm line search*, not ft1's `2e-7` (which is an ancestor anchor from the PR130
12k QAT checkpoint on mps, and whose "= upstream stage-08 tail" citation points at an `e2e.py` that is not on disk).

Because the parameter is the **code**, the LR has a unit: code units per step. AdamW's update is normalised to ≈ lr per step, so a cosine
run of T steps drifts ≈ `0.5·lr·T` code units. Setting an O(1)-code drift over the horizon gives `lr = 2/T`; at T = 3,000 that is
**6.7e-4 code units/step**.

Measured twice. 30-step CPU smoke, batch 1: predicted `0.5·6.7e-4·30 = 0.01005`, **measured 0.009124** — ratio **0.91**. 25-step MPS
smoke, batch 4: predicted `0.5·6.7e-4·25 = 0.008375`, **measured 0.008609** — ratio **1.02**. The derivation holds on this vehicle across
both devices and both batch shapes, which is the line search, done cheaply and against the actual object.

**Then re-derived against §3.** The O(1)-code target was chosen while the rate was believed expensive. The rate law says a full rewrite
of all 12,672 codes costs 176.7 B = 139 cells, so an O(1) drift is needlessly timid — at `lr = 6.7e-4` NO code crosses a rounding
boundary in 25 steps. The launch LR is therefore **2e-3** (an O(3)-code drift over 3,000 steps), and the n600 evaluation every 300 steps
IS the rest of the line search: the best-by-instrument checkpoint is kept, so an ft1-style excursion is caught at step 300 rather than at
the end. This is a constant being re-derived when its premise moved, not a knob being turned.

---

## 6. Prediction vs measured, row by row (charter §PRIOR-LAW PREDICTION)

| charter row | predicted | measured | verdict |
|---|---|---|---|
| head + one block at int4 ≈ **12 KB raw**; price replace vs coded delta | ~12,000 B | **176.7 B** for a full rewrite of all 12,672 codes (0.0139 B/code) | **superseded** — the replace form is 68× cheaper than the raw-size framing; the coded-delta form is unnecessary |
| pre-re-solve d_pose rises **10–300×** | 10–300× | not yet at scale (smoke drift 0.009 codes: stale d_pose 3e-8 … 1.5e-5 around a 5.09e-6 base) | OPEN — needs the trained delta |
| per-pair re-solve recovers **≥ 10×** | ≥ 10× | structural precondition MET (rank 6/6 on all 600, reach 1,892–2,020 codes); recovery on a GLOBAL change not yet measured | OPEN |
| post-re-solve pose leg ≤ **+2e-4 S** | ≤ 2e-4 S | OPEN | OPEN |
| instrument residual falls **8–20 %** (1,000–2,600 cells) | 8–20 % | **NOT MEASURED** — training gated | OPEN |
| *(my own)* naive-rounding pose bound as a floor | 1.288e-4 | **FALSIFIED by the incumbent** (live row is 25.3× below it on the same lattice) | withdrawn |

---

## 7. What is built (and re-runnable)

`experiments/ddm_rw1_renderer_edge_foldback.py` — one module, ten subcommands, `research_only`:

`control` · `section` · `prep` · `rate-law` · `train` · `export` · `render` · `seg` · `seg-merge` · `pose` · `admit`,
driven for the sharded stages by `experiments/ddm_rw1_admission_shards.sh` (`STAGE=render|seg|pose`, strided shards, never a prefix).

Design points worth keeping:
- the semantic section is opened by **driving the receiver's own `walk_sm3r`**, generalised from fe1's single-tensor form to a full
  run table, so every quantized tensor's `(scales, codes, bits, count, offset)` is located rather than hand-tabulated;
- editing a **row-pruned** run is refused outright;
- the candidate decode copies the even (carrier) frames and re-renders only the odd ones, because the weight delta does not touch the
  carrier — copying is exact, re-rendering would introduce a difference the candidate does not have;
- seeding that 3.66 GB copy is its **own step**, so four shards cannot race on it;
- the seg merge **refuses partial coverage** (a sub-n600 seg verdict is a toy on this axis) and the admission refuses a missing pose row;
- the admission treats the **weight delta as ONE atom** (the receiver has no per-pair weight selector, so a losing pair cannot opt out)
  while the **carrier choice stays per pair and free** (each pair ships whichever of its stale or re-solved twelve codes measures lower).

---

## 8. Verdict scope, and what is honestly still open

**No verdict is claimed on the charter's falsifiers.** Falsifier (a) — reach < 3 % after one widening — and falsifier (b) — admitted sum
≥ −2e-5 S — both require the training run, which is gated behind sj1's pass-4 receipt by MAIN's sequencing rule. Saying the reach is
"promising" would be exactly the means-as-ends failure the goal section forbids; it is simply not measured.

What IS settled at `verdict_scope: measurement` on this vehicle:

1. the rate cost of the int4-code actuator on the semantic section (§3) — **0.0139 B/code, break-even 0.011 cells/code**;
2. the structural sufficiency of the per-pair re-solve (§4) — **rank 6/6 on all 600 pairs, reach 1,892–2,020 codes**;
3. the looseness of the naive-rounding pose bound against the incumbent (§4.1) — **25.3×**;
4. the LR derivation for a code-domain actuator (§5.1) — **drift = 0.5·lr·T, measured 0.91× predicted**;
5. that trained == realized is achievable as an identity for this actuator (§1, §2).

**The pointer did not move.** The next unit is the training run and the four-stage admission behind it, aimed directly at the reach
number.

---

## 8b. THE REACH — MEASURED. The actuator is not weak; it is aimed backwards.

The gate opened, the run fired (3,000 steps, MPS, batch 4, lr 2e-3, one Metal occupant, footprint declared as a **measured 6.28 GiB
system-availability delta** — RSS accounting sees only 0.16 GiB of it, which is exactly why gov3 asks for the other unit). Speed on the
quiet machine: **0.34 s/step**, not the 2.17 s measured under load.

### The seg leg's own null control passed first

Before reading any candidate number: my n600 instrument, run on the **unmodified** decode, returns **12,866 cells** and
`d_seg_local = 0.00010906643337673611` against the pinned `0.0001090664333767361`, with **600 of 600 pairs unchanged**, in 24.8 s at four
shards (`segnull/SEG.json`). The seg leg reproduces sj1's instrument exactly.

### The device gap, measured four times for free

The first four evaluations all ran with **zero changed shadow codes** — the EMA shadow (decay `1 − 5/3000`, window 600) had not yet
crossed a rounding boundary — and all four returned **12,871 = live + 5**. So the MPS monitor's gap to the cpu_torch instrument is
**5 cells (0.039 %)**, **77× below** the 386-cell falsifier threshold. The monitor is trustworthy, and the flat prefix is a control, not a
missing measurement.

### The curve

| step | shadow codes changed | flips | vs live | reach | cells broken per changed code |
|---:|---:|---:|---:|---:|---:|
| 300 | 0 | 12,871 | +5 | −0.039 % | — (device gap) |
| 600 | 0 | 12,871 | +5 | −0.039 % | — |
| 900 | 0 | 12,871 | +5 | −0.039 % | — |
| 1,200 | 0 | 12,871 | +5 | −0.039 % | — |
| 1,500 | **4** | 13,364 | +498 | **−3.87 %** | **123** |
| 1,800 | **18** | 13,713 | +847 | **−6.58 %** | **47** |
| 2,100 | **35** | 13,992 | +1,126 | **−8.75 %** | **32** |

**The reach is negative and monotonically worsening.** This is not the charter's anticipated failure ("the edge is not reachable through
these two layers"). The edge is *extremely* reachable: **eighteen changed int4 codes move 847 argmax cells.** Compare sj1's token
pre-distortion, which repairs **1.081 cells per changed token**. The renderer-weight code actuator carries roughly **30–120× the argmax
leverage per changed symbol at 1/48th the byte cost** (0.0139 B/code vs 0.664 B/token). What it does not have is aim.

That reframes the arm's open question. It is not *can a weight change move a class edge* — measured yes, violently. It is *can any search
aim it*, against a collateral population that sj1 §16d measured at **44.5 correct boundary cells at risk per residual cell in reach**.

### The actuation, read off the checkpoints (`receipts/POSTTRAIN.json`)

| step | latent codes changed | latent max drift | shadow codes changed | shadow max drift |
|---:|---:|---:|---:|---:|
| 300 | 11 | 0.538 | 0 | 0.120 |
| 600 | 90 | 0.573 | 0 | 0.287 |
| 900 | 116 | 0.594 | 0 | 0.395 |
| 1,200 | **141** | 0.632 | 0 | 0.462 |
| 1,500 | 134 | 0.625 | 4 | 0.520 |
| 1,800 | 136 | 0.633 | 18 | 0.556 |
| 2,100 | 129 | 0.627 | 35 | 0.583 |
| 2,400 | 122 | 0.635 | 46 | 0.602 |

Two things fall out that the loss curve alone does not show.

**The optimizer is oscillating, not converging.** The latent's changed-code count PEAKS at 141 (step 1,200) and then *declines* to 122
while the max drift sits flat at ~0.63 from step 1,200 on. Codes are crossing the rounding boundary back and forth. That is a random walk
pinned near ±0.5, not a descent — and it is the same fact as the ~5 % direction persistence, seen in the parameter rather than in the loss.

**The EMA's monotone catch-up accidentally swept the dose-response curve.** Because the shadow is an exponential average converging toward
a nearly-static latent, its changed-code count rises monotonically 0 → 4 → 18 → 35 → 46 while the underlying object barely moves. Each
evaluation therefore prices a different *dose* of the same direction:

| codes changed | extra flips | marginal cells per code |
|---:|---:|---:|
| 4 | 493 | 123 |
| 18 | 847 | 47 |
| 35 | 1,126 | 32 |
| 46 | 1,238 | 27 |

Strongly **sublinear with a large intercept** — the first handful of codes does most of the damage and further codes add little. That
shape says the damage is not an accumulation of independent local edits; it is one global photometric shift of the rendered frames that
the first few head/`blocks.3.pw` codes already impose. 119 of the 122 changed codes sit in `blocks.3.pw.weight` (73 % of the code budget,
so a mild 1.34× over-representation) — the pointwise mixer, not the head.

### The confound this measurement names about itself

At batch 4 the fixed-τ surrogate has **no trend** over 1,300 steps (first-10 mean 4.796e-4, last-10 4.976e-4) and the code drift
**saturates at ~0.6** against a 3.0 free-drift bound — the per-step direction is only ~5 % persistent. So the negative reach is
**not yet interpretable** between two readings that prescribe opposite next moves:

1. **no descent direction exists** in this code subspace (the actuator is closed at this formulation), or
2. **the direction exists and the minibatch draw noise buries it** (the search is at fault, not the actuator).

Two controls separate them, both cheap and both built: `perturb-control` prices a RANDOM ±1 code change of the same size (if random costs
about the same, the surrogate is not steering at all), and `--full-field` accumulates the exact n600 gradient over all 600 pairs in 150
chunks (~50 s/step) so draw noise stops being a candidate explanation. With 12,672 parameters and a deterministic objective, the
noise-free gradient is affordable — which is the honest answer to a saturating minibatch drift.

### The random-direction control settles the confound (`receipts/PERTURB.json`)

Random ±1 code changes of the same sizes, two seeded draws each, same n600 realized-argmax path, same null
(`null_flips_same_path = 12,871`, the +5 device gap):

| codes changed | RANDOM extra flips (2 draws) | TRAINED extra flips | trained is better by | collateral avoided |
|---:|---|---:|---:|---:|
| 4 | 640, 1,463 (mean **1,051**) | **493** | **2.13×** | 53.1 % |
| 18 | 1,707, 2,799 (mean **2,253**) | **847** | **2.66×** | 62.4 % |
| 66 | 6,418, 5,576 (mean **5,997**) | **1,463** | **4.10×** | **75.6 %** |

**The surrogate IS steering, and its advantage GROWS with the dose** — 2.13× → 2.66× → 4.10×. So the negative reach is NOT "no descent
direction exists": a random direction of the same size is 4.1× more destructive, and the trained direction avoids **75.6 %** of the
collateral a random one inflicts.

That converts the arm's verdict from a closure into a **quantified gap**. To break even the search must avoid **100 %** of the
collateral; it reaches 75.6 % at 66 codes and is still improving with dose. The remaining 1,463 cells are the whole distance between this
formulation and a candidate. And because the rate is nearly free (66 codes = 17.9 B = 1.19e-05 S), a search that reached ~100 % avoidance
and then repaired even a few hundred cells would win comfortably.

### A design fault the full-field probe exposed on its first step: AdamW is the wrong optimizer for this actuator

The exact-gradient run's very first step moves the latent by **0.0400 code units — exactly `lr`** — and its second step reports a
byte-identical loss, surrogate, barrier and reach. Both facts are the same fact. The forward is piecewise constant in the latent (the
STE's round only changes the object when a code crosses ±0.5), and **AdamW normalises per parameter**, so *every* code with a non-zero
gradient drifts at the same ~`lr` rate regardless of how much that code matters.

That is precisely the wrong dynamics here. The rate law and the collateral measurement both say the same thing: this actuator wants
**FEW, WELL-CHOSEN** codes to move. A per-parameter-normalised optimiser instead marches all 12,672 toward the rounding boundary together,
so they cross in a near-simultaneous wave — the sparsity that makes the rate nearly free and keeps the collateral small is destroyed by
the update rule, not by the objective. Plain SGD moves each code in proportion to its own gradient and preserves exactly that sparsity;
so does an explicit top-k selection on the accumulated gradient.

This was not visible in the minibatch run because there the drift saturated at ~0.63 and only ~130 codes ever crossed — the noise was
masking the optimizer's own bias toward density. It is recorded here as a named next-arm change, not as a post-hoc excuse: the run that
exposed it was launched to answer a different question.

### The full-field probe, stopped at step 7 on purpose

It was launched to remove minibatch draw noise as an explanation, and it delivered that answer on its first two steps rather than its
last: step 1 moved the latent by exactly `lr`, step 2 reported a byte-identical loss, and those two facts together are the AdamW-density
finding above. Its remaining 18 steps would have priced a **dense** move — every code crossing the rounding boundary in a near-
simultaneous wave — which the same finding says is the wrong shape for this actuator. It was stopped at step 7 and its checkpoints and
log retained; `gradient-topk` answers the arm's question unconfounded and in a sixth of the time. Recording the stop and its reason here
because a run that is quietly abandoned is indistinguishable from one that failed.

### `gradient-topk` — the decisive probe, and it names the binding constraint

One exact n600 gradient of the seg surrogate (35.6 s), then realized flips for the k codes with the largest `|dL/dcode|`, each moved ONE
step down its own gradient. No optimizer, no schedule, no EMA, no draw noise (`receipts/GRADIENT_TOPK.json`; same null 12,871):

| k | changed codes | flips | vs null | cells broken **per code** | rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 15,785 | **+2,914** | 2,914 | 8.2 B |
| 4 | 4 | 20,726 | +7,855 | 1,964 | 8.6 B |
| 16 | 16 | 43,497 | +30,626 | 1,914 | 10.4 B |
| 64 | 64 | 237,830 | +224,959 | 3,515 | 17.6 B |
| 256 | 256 | 676,876 | +664,005 | 2,594 | 46.4 B |
| 1,024 | 1,024 | 57,097,199 | +57,084,328 | 55,746 | 161.6 B |

**Moving the single most-influential code one step costs 2,914 cells.** Put the three selection rules side by side, all in the same unit:

| how the codes were chosen | cells broken per changed code |
|---|---:|
| **top-`|gradient|`**, one step | **1,914 – 55,746** |
| random ±1 | 90 – 160 |
| AdamW-accumulated (the 3,000-step run) | **27 – 123** |

**The gradient's magnitude is ANTI-correlated with usefulness at the int4 step size** — the highest-leverage codes are 20–35× *worse*
than random to move, and AdamW beat random 2–4× precisely because its slow accumulation selects codes with SMALL, persistent gradients,
i.e. the low-leverage ones. Every number in this arm now falls out of one mechanism.

**The binding constraint is the quantization step, not the layer set and not the search.** `∂L/∂code` is a first-order quantity, valid for
infinitesimal moves. The smallest representable move here is ±1 code = one full fp16 scale unit, which for `head.weight` is **1/7 of that
weight's own dynamic range** (codes span −7…+7). The linearisation is invalid at that step size, and it is *most* invalid exactly where
the gradient is largest. The gradient is also fully dense — all 12,672 codes have non-zero gradient, `|max| 2.18e-06` against
`|median| 3.31e-08`, a 66× spread — so there is no sparse subset for the first-order model to be right about.

That is a closure at `verdict_scope: formulation` **that names its own two cures**, neither of which is the charter's widening. Its
significance at the operating point, stated rather than assumed: the residual is 12,866 cells = **0.010907 S = 57.9 % of the remaining
0.018823 gap to 0.12**, and the rate this actuator would pay for a full rewrite is 176.7 B = 1.177e-04 S = **0.63 % of that gap**. The
stake is enormous and the toll is negligible; what fails is the aim.

1. **A finer grid on these tensors.** Depth 4 → 5 or 6 on `head.weight` / `blocks.3.*` shrinks the step to 1/2 or 1/4 and would bring the
   linearisation back into range. The receiver already supports per-tensor depths 2–8 (`_decode_depth_nibbles`), so this is a packer
   change, not a receiver change. **Its rate is NOT yet priced**: §3's law prices CHANGED codes at fixed depth, and a depth change alters
   the run length — that must be measured by a real encode before it is claimed, and is registered as OWED, not as a result.
2. **Selection by realized Δflips instead of by gradient** — sj1's own method, applied to codes rather than tokens. The mechanism is
   proven on this object (32 pointer moves); the actuator is 30–120× more leveraged per symbol and 48× cheaper per symbol. The cost is a
   search: 12,672 codes × 2 directions with realized acceptance, which needs a screening subset before n600 confirmation.

### The ranking probe: `abs_asc` is the best rule found, and it still does not repair

Same probe, ranking INVERTED — the codes with the SMALLEST non-zero `|dL/dcode|` first, which is what the AdamW run's own behaviour
implied (it beat random by selecting small, persistent gradients). Pre-registered prediction, written before the run: **−1 to +30 cells
per code**, i.e. at or below the random band.

| k | changed codes | vs null | cells per code | ΔS | share of the 0.018823 gap |
|---:|---:|---:|---:|---:|---:|
| 16 | 16 | +1,203 | 75.2 | +1.020e-03 | **+5.42 %** |
| 64 | 64 | +2,090 | 32.7 | +1.772e-03 | **+9.41 %** |
| 256 | 255 | +3,648 | 14.3 | +3.092e-03 | **+16.43 %** |
| 1,024 | 1,018 | +8,204 | 8.06 | +6.955e-03 | **+36.95 %** |

**The prediction is falsified on the low side too.** `abs_asc` is comfortably the best rule found — 8.1 cells/code at k=1,024 against
random's 90–160 and `abs_desc`'s 1,914–55,746, a **238×** spread between the two ends of the same gradient — and the per-code cost falls
monotonically with dose. But it never crosses zero: the total still grows, +8,204 cells at k=1,024.

So the complete ranking picture, all in cells broken per changed code:

| rule | best measured |
|---|---:|
| top-`|gradient|` (`abs_desc`) | 1,914 |
| random ±1 | ~90 |
| AdamW-accumulated | 27 |
| bottom-`|gradient|` (`abs_asc`) | **8.1** |
| **needed to break even** | **≤ 0 (repair)** |

**Not one selection rule repairs anything.** Every dose of every ranking increases the residual. The gradient carries a real and strong
ORDERING — 238× between its ends — but the ordering is over *how much damage a full int4 step does*, not over *which move helps*.

### The discrete realized search: the actuator's minimum quantum of action is measured

MAIN's directive, and the honest last card: propose ±1 groups down the surrogate's sign, render, score with the frozen SegNet, **accept
only if realized flips strictly FALL**, bisect a rejected group so a good move is not thrown away with a bad one. Screened on 120 seeded
random pairs (null 2,646 flips), confirmed at n600. Pre-registered before launch (`receipts/PREREG_DISCRETE_SEARCH.json`): expected 120 /
340 / 1,500 cells repaired at 66 / 200 / 1,000 accepted codes; falsifier at < 139 cells net.

The search bisected 128 → 64 → 32 → 16 → 8 → 4 → 2 → **1** and kept going to its evaluation budget. Final
(`receipts/CODE_SEARCH.json`, 660 s):

| | |
|---|---|
| evaluations | **150** |
| accepts | **0** |
| codes changed | **0** |
| screen best vs screen null | 2,646 vs 2,646 (**unmoved**) |
| n600 final vs n600 null | 12,871 vs 12,871 (**unmoved**) |
| cells repaired at n600 | **0** |
| ΔS | **0.0** (0.00 % of the gap) |
| pre-registered falsifier (< 139 cells) | **FIRED** |

The best move of any size found anywhere in the sweep was **+48 flips on the 120-pair screen** (≈ +240 at n600). The search returned the
base object exactly — which is the property it was built for: realized acceptance cannot lose, so a zero here is a clean negative and not
a damaged candidate.

That is the arm's closing measurement, and it is a statement about the REPRESENTATION rather than about any search:

> **The smallest action this actuator can take — one int4 code, one step, chosen from the cheapest end of the gradient — already costs
> on the order of 240–455 flipped cells. There is nothing smaller in the representation to try.**

A search cannot find a move below the grid's own resolution. That is why every rule fails, why `abs_asc` beats `abs_desc` by 238× and
still never crosses zero, and why AdamW's slow accumulation looked better than a single decisive step: it was approximating a *smaller*
move than the grid can express, right up until the round made it express a whole one.

**Scored honestly at the operating point** (S 0.13882326433317044, gap to 0.12 = 0.018823264): the residual this arm set out to repair is
12,866 cells = **0.010907 S = 57.9 % of the whole remaining gap**. Nothing in this arm was dismissed as small — that stake is the reason
the arm exists. What is closed is one FORMULATION for reaching it.

## 8c. Counting falsifier (a) plainly

The charter's falsifier (a): *"after 3,000 steps at the object's own LR the instrument residual falls < 3 % → widen to all four blocks
ONCE at the same LR; if still < 3 %, the renderer-weight actuator is closed on this object (formulation scope)."*

**It fired.** The residual did not fall 3 %; it ROSE, monotonically, to −9.6 % at 46 changed codes. Counted plainly: this formulation —
head + `blocks.3` int4 codes, expected-flip surrogate at annealed τ, minibatch 4, AdamW at lr 2e-3 with the S-linearised pose terms —
**does not repair the residual. It damages it.** No candidate was built, no admission was run, and no seal exists, because a candidate
that is worse on seg cannot become better on pose or rate: `dS_seg` alone is +1,238 × 8.477e-07 = **+1.05e-03 S**, which is 52× the
admission bar in the wrong direction.

**What the charter's prescribed cure does NOT fit.** The widening exists to test "the edge is not reachable through these two layers."
The measurement says the opposite: the edge is violently reachable — 18 codes move 847 argmax cells. Adding `blocks.2` at the same LR adds
10,080 more codes to a search that is already oscillating at ~5 % direction persistence; it would spend an hour to answer a question this
data did not ask, and its most likely result is a larger negative. Recording that the prescribed cure is mis-aimed is not a licence to
skip it — it is the reason to run the two controls FIRST, because they decide whether widening is even the right axis.

**What the data prescribes instead**, in order:

1. **`perturb-control`** — **RUN, and it answered**: random costs 2.1–4.1× more, so the surrogate IS steering and the finding is about
   the collateral ratio, not about the existence of a direction (see the table above).
2. **`--full-field`** — **RUN and STOPPED at step 7**, having answered on its first two steps: AdamW normalises per parameter, so it
   marches every code at one rate and destroys the sparsity this actuator needs.
3. **`gradient-topk`** — **RUN, and it is the decisive one**: the first-order model is invalid at the int4 step size, most invalid where
   the gradient is largest. The actuator is closed at `verdict_scope: formulation`, and the charter's widening is NOT the cure — a finer
   grid or a realized-acceptance search is.

**What is NOT closed by this.** The paradigm — *a renderer-weight change admitted per pair through the carrier re-solve* — is untouched.
Two of its three legs measured favourably before the search failed: the rate leg is nearly free (0.0139 B/code, break-even 0.011 cells per
code) and the pose leg's structural precondition holds on every pair (rank 6/6, reach 1,892–2,020 codes). The leg that failed is the
one that picks WHICH codes to move. Per Catalog #307 this is an IMPLEMENTATION-level falsification with the paradigm intact.

## 8d. Door (1): the per-row fp16 SCALES — and the exponent that closes door (2) by arithmetic

The closing law said the int4 grid is the binding constraint, so the first door is the actuator whose step is FINER.
`weight[i,j] = code[i,j] · scale[i]`, so moving one per-row fp16 scale by k ULPs moves a whole row by a **code-proportional fraction** of
a code step. Measured before launch (`receipts/PREREG_SCALE_SEARCH.json`), in units of one code move:

| tensor | rows | weights/row | one 1-ULP move |
|---|---:|---:|---:|
| `head.weight` | 3 | 864 | 1.342 (**coarser** — excluded from the pool by measurement) |
| `blocks.3.dw.weight` | 96 | 9 | **0.0242** — a **41× finer** minimum action |
| `blocks.3.pw.weight` | 96 | 96 | **0.141** — 7.1× finer |

The seam has its own null control (shipped scales in → byte-identical 36,130 B body and 31,792 B stream), the walk is on the fp16 grid
itself (bit-pattern ULPs: monotone, exact, reversible — a move that is not a whole ULP is not a move the receiver can express), and the
container search prices ONE changed scale at **−1 byte**, so the rate break-even is *negative* and any repair at all would win.

### What 91 realized evaluations measured

| tensor | ULP | n | min | median | max |
|---|---:|---:|---:|---:|---:|
| `blocks.3.dw` | ±1 | 24 | **7** | 14–15 | 29 |
| `blocks.3.dw` | ±2 | 23 | 10 | 21–22 | 36 |
| `blocks.3.pw` | ±1 | 22 | **6** | 16–17 | 28 |
| `blocks.3.pw` | ±2 | 22 | 16 | 21–24 | 38 |

(screen deltas on 120 pairs; ×5 for n600). **Zero accepts** — over the full 400-evaluation run as well as this first 91.

**The pre-registered proportionality is FALSIFIED.** `dw` perturbs 5.8× less than `pw` and costs the *same*; one dw ULP perturbs **41×**
less than one int4 code and costs only ~4.8× less. Fitting the exponent across every pair of measured scales:

| comparison | exponent |
|---|---:|
| one int4 code vs `pw` 1-ULP | 0.545 |
| one int4 code vs `dw` 1-ULP | 0.322 |
| `pw` 1-ULP vs `dw` 1-ULP | **0.073** |
| `dw` 1 → 2 ULP | 0.568 |
| `pw` 1 → 2 ULP | 0.447 |

**Damage grows roughly as the SQUARE ROOT of the perturbation, and flattens further at the fine end.** That is the signature of a
knife-edge population — cells sitting at essentially zero SegNet margin that flip under any nudge at all, which is exactly sj1 §16's
picture (99.67 % of the residual lies on a GT class edge, with 44.5 correct boundary cells at risk per residual cell).

### Door (1) final (`receipts/SCALE_SEARCH.json`, 1,729 s)

| | |
|---|---|
| proposals available | 768 (192 rows × 4 ULP steps) |
| evaluations spent | **400** (52 % of the pool) |
| accepts | **0** |
| scales changed | **0** |
| screen best vs null | 2,646 vs 2,646 (**unmoved**) |
| n600 final vs null | 12,871 vs 12,871 (**unmoved**) |
| ΔS | **0.0** — **0.00 %** of the gap |

**A correction on my own falsifier, before anyone reads the flag.** The receipt says `falsifier_fired: false`, and that is a **vacuous**
false, not a pass: the pre-registered condition was *"the accepted set repairs fewer cells than the rate break-even for the scales it
changes"*, and with an EMPTY accepted set the rate is 0 B, the break-even is 0.0 cells, and `0 < 0.0` is false by arithmetic rather than
by evidence ([[m50]] — vacuity reads as PASS). **The verdict is the zero itself**: 400 realized evaluations of the finest actuator the
archive can express, with a negative rate toll, accepted nothing. Door (1) is closed at `verdict_scope: formulation`.

### Door (2) is closed by that exponent, before it is built

At exponent 0.5, taking the minimum damage from **~72 cells** down to **1** needs a perturbation reduction of **5,184× = 12.3 extra
bits**. An int4 → depth-5/6 change buys **1 or 2** bits (2–4×), which moves 72 cells to **51 or 36** — still far above any break-even.
So the depth arithmetic does **NOT** clear, and pricing a depth change by real encode would be spending on a lever the exponent already
refutes. Recorded as an arithmetic closure, not an untried option.

The same arithmetic, run forward, names the one place it *does* clear: **fp32 per-row scales** are +13 mantissa bits = 8,192× finer,
predicting a minimum damage of **0.8 cells** — below the point where a steered search can plausibly net a repair. The cost is +2 B per row
× 195 rows = **+390 B = 2.597e-04 S = 1.38 % of the gap**, and it needs a RECEIVER change (the parser reads `<f2`), so it is a different
arm's charter, not a knob here.

## 8e. OWED #1e sizing — the damage curve does NOT floor, and the window is 6 bits wide, not 13

Before a line of receiver code: the premise the whole fp32 chain rests on — that damage keeps falling below the fp16 step — is a property
of the **render**, not of the archive format. A sub-fp16 scale move is expressible in the float32 forward; only *shipping* it needs a
receiver change. So it was measured with zero receiver work (`receipts/SCALE_SIZING.json`, 135 s, `blocks.3.dw`, 6 seeded random rows ×
both signs, seeded random 60-pair screen):

| relative scale move | × below the fp16 ULP | median cells (n600-scaled) | min |
|---|---:|---:|---:|
| 7.09e-04 (the fp16 ULP) | 1.0 | **25.0** | 10 |
| 1e-04 | 7.1 | **10.0** | 0 |
| **1e-05** | **70.9** | **0.0** | 0 |
| 1e-06 | 709 | 0.0 | 0 |
| 1e-07 | 7,090 | 0.0 | 0 |

**The pre-registered falsifier did NOT fire.** It said *"if the median damage at rel = 1e-05 has not fallen below 36 cells at n600, a
floor exists and fp32 scales are refuted."* Measured: **0**, with a max of 0 across all twelve probes. The fitted exponent on this sweep
alone is **0.468**, and it predicts **0.31 cells** at the fp32 ULP. There is no damage floor — the fp16 grid was simply too coarse, which
is exactly what OWED #1e was written against.

**But zero damage is also zero effect**, and that is what the sweep really buys. A move that changes nothing cannot repair anything, so
the useful regime is the window where a move changes *something* and can net negative:

> **rel ∈ [1e-05, 7.09e-04] — a factor of 70.9, i.e. 6.15 extra mantissa bits, NOT 13.**

That re-prices the format change by 2.5×:

| extra bits/row | reaches rel | +bytes | ΔS | share of gap | break-even |
|---:|---:|---:|---:|---:|---:|
| **+3** | 8.9e-05 | **+73.1 B** | 4.87e-05 | **0.26 %** | **57 cells** |
| +4 | 4.4e-05 | +97.5 B | 6.49e-05 | 0.34 % | 77 cells |
| +6 | 1.1e-05 | +146.2 B | 9.74e-05 | 0.52 % | 115 cells |
| +13 (full fp32) | 8.7e-08 | +316.9 B | 2.11e-04 | 1.12 % | 249 cells |

So the receiver change to aim for is **+3 to +6 mantissa bits per row**, not fp32 — a third of the bytes, and the extra bits below
rel = 1e-05 buy nothing because the render stops responding there.

**A sampling note that earned itself.** The seeded random 60-pair draw nulls at **1,254** flips; the first-60 **prefix** nulls at
**1,080** — the prefix is **13.9 % easier**, a stronger seg prefix bias than the 0.95–0.97× in memory. The verdict is taken on the random
draw; the prefix number is reported because it was asked for, and labelled.

## 9. OWED (the queue this arm hands forward, each with its blocker named)

- **OWED #1 — the reach. DONE, and it is negative** (§8b). Run, measured, counted. No further work owed on this row.
- **OWED #1c — the finer grid (int4 → depth 5/6): CLOSED BY ARITHMETIC, not by a build.** §8d fitted the damage exponent at
  **0.073–0.568** across every measured perturbation scale; at 0.5, moving the minimum damage from 72 cells to 1 needs **12.3 extra
  bits** and a depth change buys **1–2**, leaving 51 or 36 cells — still far above any break-even. Pricing it by real encode would be
  spending on a lever the exponent refutes. STATUS: `closed-by-derivation`.
- **OWED #1e — fp32 per-row scales, the one place the arithmetic DOES clear.** +13 mantissa bits = 8,192× finer ⇒ predicted minimum
  damage **0.8 cells**, below where a steered search can plausibly net a repair. Cost +2 B × 195 rows = **+390 B = 2.597e-04 S = 1.38 %
  of the gap**, and it needs a RECEIVER change (the parser reads `<f2`). STATUS: `genuinely-deferred-because-receiver-change`; it is a
  different arm's charter, and the arithmetic above is what that charter should be written against.
- **OWED #1c-superseded — the finer grid as originally framed.** The receiver already supports per-tensor depths 2–8
  (`ddm_mp2_semantic_receiver._decode_depth_nibbles`), so depth 4 → 5 or 6 on `head.weight` / `blocks.3.{dw,pw}.weight` is a PACKER
  change, not a receiver change, and it halves or quarters the minimum step. Two things must be measured before anything is claimed:
  (a) the RATE of a depth change — §3's law prices *changed* codes at *fixed* depth, and a depth change alters the run length, so it
  needs a real encode (2,592 + 864 + 9,216 codes × 1 extra bit = 1,584 B raw before compression); and (b) whether the damage scales
  linearly with the step. If it does, quartering the step takes the cheapest single move from ~240 cells to ~60 — still positive, which
  would mean depth 8+ or nothing. If the 238× spread between the gradient's ends reflects a superlinear response, a half-step could land
  much better. Neither is known. STATUS: `READY`, `EV: high` — one packer change and one probe.
- **OWED #1d — the fp16 SCALES: RUN, and closed at formulation scope** (§8d). 41× finer minimum action, negative rate break-even,
  91+ realized evaluations, **zero accepts**, floor 6–7 cells on the screen. The actuator is real and the toll is negative; what fails
  is that a knife-edge population flips under any nudge. Superseded description follows for provenance.
- **OWED #1d-superseded — the same actuator with a SMALLER-THAN-GRID move.** The one thing the int4 grid forbids is a fractional step, but the
  archive does not: a per-row fp16 SCALE change moves every code in that row by a fraction of a step, and the scales are already
  shipped (390 B across the three tensors). A scale is a continuous knob over a whole row — coarser in reach, finer in amplitude — and it
  is the exact complement of what failed here. Untried. STATUS: `READY`, `EV: high`, and cheap: it reuses this arm's whole chain.
- **OWED #1b — close the 24 % collateral gap.** The control (§8b) says the surrogate avoids 75.6 % of a random direction's damage and
  needs 100 %. Live probe: the exact-n600-gradient run (`--full-field`, 25 steps at lr 0.04, ~21 min) removes the ~5 %-persistence draw
  noise as an explanation. RUN — and answered, along with `gradient-topk` in both ranking directions and the discrete realized search.
  All four say the same thing: the binding constraint is the int4 step, not the noise and not the layer set. STATUS: `DONE`.
- **OWED #2 — the per-pair recovery on a GLOBAL change.** `FIRE_ORDER` step 4c. Nobody has measured whether the per-pair re-solve
  keeps fe1's 643–3,053× when all 600 renders move at once; §4 shows the structural precondition holds on every pair. It was NOT run
  here because a candidate that is worse on seg cannot be rescued on pose, so spending 1.4 h on it would have been means-hoarding.
  STATUS: `blocked-by-a-candidate-that-is-not-worse-on-seg`.
- **OWED #3 — the widening, only if falsifier (a) fires.** `--widened` opens `blocks.2.{dw,pw}.weight` (+10,080 codes, still
  depth 4 and un-pruned). At the measured rate law that is still ~+200 B for a total rewrite. STATUS: conditional.
- **OWED #4 — the `blocks.*.film` half is out of scope by construction**, because those runs are row-pruned to 2 of 192 rows and
  editing them would re-open the ft1 realization gap. Whether a FiLM fold-back is reachable at all needs a receiver change
  (a differently-pruned SM3R), which is a different arm. STATUS: `genuinely-deferred-because-row-pruned-run-is-not-byte-expressible`.

---

## 10. Equations leg

Two laws carry this arm forward, both registered in `tac.canonical_equations`:

- **`model_section_edit_container_break_fee_v1`** — third anchor
  `rw1_semantic_int4_weight_code_rewrite_marginal_0p0139_bytes_per_code_20260909`, closing that law's own excluded clause
  ("extrapolation past N = 200 changed codes") on a different run. The container-break fee SATURATES: 0.0139 B/code at N = 12,672.
- **`renderer_edge_layer_foldback_reach_v1`** (NEW) — the closing law: *one int4 code step in `head`/`blocks.3` already moves 240–455
  argmax cells, so a renderer-weight fold-back is limited by the grid's resolution, not by the search.* `verdict_scope: formulation`;
  its `excluded` list names the two live doors (a finer grid, and the per-row fp16 scales) so the next arm inherits the opening, not
  just the closure. Anchor `rw1_int4_code_minimum_action_costs_240_to_455_cells_20260909`, residual: both predictions falsified, the
  measured reach has the OPPOSITE sign to the charter's.

---

Own-vehicle frontier: **S 0.13885056455024844 @ 181,414 B [contest-CUDA T4 n600]**, archive sha `c810c2c7…f671e` (rc2, 33rd move).

### MAIN note (2026-09-09 17:40Z) — relative significance of the step-7 stop, and the verdict's scope

The full-field probe's stop at step 7 is an INSTANCE verdict on a DIAGNOSTIC (it answered "is the oscillation minibatch draw noise?" — no: step 1 moves the latent by exactly lr and step 2 is byte-identical, so the forward is piecewise constant in the latent), not a magnitude dismissal of the lever. The lever's stake, stated at the current operating point: the residual this arm aims at is 12,443 cells = 100·12,443/(600·196,608) = **0.010547 S = 56.0 % of the remaining gap 0.01882326** (S 0.13882326433317044 → 0.12). Nothing of that stake is dismissed by the stop. What the arm MEASURED instead: the trained direction beats random ±1 code changes by 2.13×/2.66×/4.10× at 4/18/66 codes (growing with dose); 18 codes move 847 argmax cells; 66 codes cost 17.9 B = 1.19e-05 S = 0.063 % of the gap. verdict_scope: FORMULATION (AdamW on a continuous latent over a piecewise-constant forward) — the discrete realized search over the same 12,672 codes, guided by the measured surrogate direction, is the open successor and is not closed by this row.
