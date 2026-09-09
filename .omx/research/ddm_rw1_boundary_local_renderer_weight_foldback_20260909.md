# ddm_rw1 — the joint renderer fold-back on the shipped object: the actuator is the int4 CODE, the rate corner is open at 0.014 B/code, and the incumbent falsified my own closed form before any training step (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm `ddm_rw1` · Lane `lane_ddm_rw1_renderer_edge_foldback_20260909` · Charter
`.omx/research/charters/ddm_rw1_boundary_local_renderer_weight_foldback_per_pair_admitted_20260909.md` · Gestalt: gs3 Addendum 15.

Axes: d_seg `[macOS-CPU advisory, cpu_torch argmax, DALI lineage, n600]` · d_pose `[cpu_torch fp32 authority, DALI GT, n600]` ·
training `[macOS-MPS research-signal]` · bytes EXACT (real container-searched encode) · `score_claim=false` throughout; no T4 row exists
for this arm and MAIN fires.

---

## 0. Headline

**The pointer did NOT move on this arm.** What landed is the instrument, three passing identity controls, and one measurement that
inverts the charter's own rate premise:

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

The number the arm exists to buy — the **reach** of the renderer-weight actuator on this object — is **NOT YET MEASURED**. Training is
gated behind sj1's pass-4 receipt (MAIN's sequencing rule: its five CPU shards starve a Metal cell's host thread). Everything upstream of
that gate is built, controlled and committed.

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

Measured in the 30-step CPU smoke: predicted drift `0.5·6.7e-4·30 = 0.01005`, **measured 0.009124** — ratio **0.91**. The derivation
holds on this vehicle to 9 %, which is the line search, done cheaply and against the actual object.

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

Own-vehicle frontier: **S 0.13885056455024844 @ 181,414 B [contest-CUDA T4 n600]**, archive sha `c810c2c7…f671e` (rc2, 33rd move).
