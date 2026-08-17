---
arm: ddm_ce1
title: "the +49,580-flip opening excursion is the CE objective's DIRECTION, and the curriculum is ordered backwards: 81.2% of the LR budget is spent on the WORST-aligned objective (cos_sign 0.209) and 0.84% on the BEST (0.619) -- a split that is SCALE-INVARIANT, which is why 5x the window changed nothing; four of five candidate mechanisms are refuted by measurement"
utc: 2026-08-17
charter: "ddm_ce1 -- WHY does CE create +49,580 flips in 100 steps? (the named seg successor, $0)"
axis: "[macOS-MPS training-signal] re-analysis of already-retained payloads + [macOS-CPU advisory] checkpoint forensics. NO training ran. NEVER a score."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-finding INSTANCE on the named retained runs; the alignment-weighted-LR-budget scale-invariance is DERIVED with its closed form and verified numerically at three window lengths"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ce1 — the opening excursion is a direction, not a shock

**STORES CONSULTED (read at source, never from a summary):**
`.omx/research/ddm_l3000_no_descent_verdict_20260817.md` (sha `83025c4e…`, commit `eb13f7e108`) ·
`.omx/research/ddm_rg1b_band_objective_build_20260816.md` §2.7, §6.2 (sha `e1306299…`) ·
`.omx/research/ddm_wallclock_prefix_bias_law_20260817.md` (sha `ee3e061b…`) ·
`src/tac/pr130_lift/lifted/semantic_renderer_oracle.py` (sha `ffdf0988…` — **byte-identical to the
file rg1b measured against**) · `src/tac/pr130_lift/train_semantic_quantized_resumable.py`
(sha `b486f416…`) · retained payloads `/Volumes/APDataStore/pact/ddm_jr1/{A2_repeat,L3000_off}/` and
`/Volumes/APDataStore/pact/ddm_lr1/{C0,A1,A3,W1}/` (READ-ONLY; nothing written there).

---

## ANSWER FIRST

**The CE phase's gradient points ~73–84° away from the metric's own gradient, and the curriculum runs
it FIRST, for HALF the run, from the best starting point the run will ever have.** The +49,580 flips
are not a shock, not a re-anchor, and not an EMA artifact. They are what it costs to move **0.064% of
the weight norm** in a direction that is nearly orthogonal to the thing being measured.

**Four of the five candidate mechanisms are refuted by measurement, not by argument:**

| # | candidate | verdict | the measurement that settles it |
|---|---|---|---|
| 1 | LR warm-up transient | **REFUTED** | There is no ramp. `CosineAnnealingLR` only; lr is at **99.73%** of peak at step 100; `--float-warmup-steps 0`. |
| 2 | CE anti-aligned with the argmax metric | **CONFIRMED** | rg1b §6.2 real gradient path at the init: CE `cos(sign g) = 0.2087` (73.5° raw) vs softplus 0.5235 (45.4°) vs expected_flip 0.6185 (35.9°). Plus the three controls below. |
| 3 | QAT scale re-anchor at step 1 | **REFUTED** | Opening displacement is structurally *identical* to late displacement: top-3 frac_sq 0.5384 (init→A2@100) vs 0.4579 (A2@100→A2@600), same top tensors, smoothly graded across all 38. A re-anchor would concentrate; it does not. |
| 4 | EMA / evaluated-weights basis | **REFUTED** | Shadow is within 4% of live in norm and cos ≥ 0.997 throughout. Both runs sit in the warm-up branch `min(target,(1+t)/(10+t))`, so effective decay is **identical** (0.91818 at t=100). |
| 5 | Adam early transient | **REFUTED** | Saturation `r = |m̂|/(√v̂+ε)` is **0.0846** at t=100, at or *below* its later value 0.1779 at t=300. The early effective step is not inflated. |

**And the deep finding, which explains the whole jr1/L3000 program in one line:**

| phase | steps (T=3000) | % of LR budget | `cos(sign g)` @init | aligned budget |
|---|---|---:|---:|---:|
| `ce` | 1–1500 | **81.19%** | **0.2087** | 16.9% |
| `softplus_margin` | 1501–2550 | 17.97% | 0.5235 | 9.4% |
| `expected_flip` | 2551–3000 | **0.84%** | **0.6185** | 0.5% |
| **total** | | 100% | | **26.87%** |

**The curriculum spends 81.2% of its learning rate on its worst-aligned objective and 0.84% on its
best.** And because `ce_fraction`, `softplus_fraction`, and `CosineAnnealingLR(T_max)` are *all*
fractions of the run, this split is a **SCALE-INVARIANT** of the trainer — verified numerically at
T = 600 / 3000 / 30000: `81.15 / 81.19 / 81.20%` and aligned budget `26.88 / 26.87 / 26.87%`.

**That is why 5× the window changed nothing, and why 50× would not either.** L3000 was not a
truncation and not an asymptote-of-optimization — it was the *same schedule*, re-parameterised.
Lengthening a run cannot change a quantity that is defined as a fraction of the run.

**Pointer UNMOVED.** No training ran, no archive was built, no score was produced. This unit is MEANS.

---

## §1 The mechanism, at source

`src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:181-193`, sha `ffdf0988…`:

```python
progress = step / max(total_steps - 1, 1)
if progress < ce_fraction:
    temp = 1.0 * (0.08 ** (progress / ce_fraction))
    return F.cross_entropy(logits / temp, target), "ce"
```

The CE phase divides the logits by a temperature annealing **1.0 → 0.08** across the phase. At
`temp = 1.0` the softmax is soft, so the ~**99.97%** of pixels that are already argmax-correct (init
`d_seg = 2.86e-4`) still carry gradient. rg1b measured the consequence exactly: the stock objective
puts **2.161%** of its gradient mass on the 1-px label band, which is **2.157%** of the pixels — ratio
**1.0016**, *exactly area-proportional* — against a debt that is **99.22%** on-band. The temperature
is the knob that sets that allocation, and the anneal starts at the setting that maximises the
misallocation.

**Correction to the charter:** the charter said the `history` rows carry `loss` / `segmentation_loss`
/ `lr`. They do not — `result.json::history` carries only `{step, quantized_exact_seg,
normalized_rgb_mse, evaluated_weights}`. The per-step loss and lr live in `run.log`. Every trajectory
below is read from `run.log`.

---

## §2 Control A — the progress clock, not the step clock

Same init, same seed, same peak lr. At matched **step** the two runs disagree by up to **3.8×**. At
matched **progress** (⇒ matched temperature) they agree to **13–16%** in the early phase.

| progress | temp | A2 step | A2 flips | L3k step | L3k flips | flip L/A |
|---:|---:|---:|---:|---:|---:|---:|
| 0.16528 | 0.4339 | 100 | 27,098 | 500 | 30,704 | **1.133** |
| 0.33222 | 0.1867 | 200 | 12,460 | 1000 | 14,429 | **1.158** |
| 0.49917 | 0.0803 | 300 | 11,146 | 1500 | 6,510 | 0.584 |

versus the naive comparison:

| step | A2 temp | L3k temp | A2 flips | L3k flips | flip L/A |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.4339 | 0.8464 | 27,098 | 49,580 | **1.830** |
| 200 | 0.1867 | 0.7152 | 12,460 | 45,901 | **3.684** |
| 300 | 0.0803 | 0.6043 | 11,146 | 41,326 | **3.708** |

The same collapse holds on the loss (matched step 2.29–3.83×; matched progress 0.78–0.90×), which
also means **the logged CE loss curve cannot be read as descent** — its instrument (the temperature)
changes at every step. Sister of the standing law that an instrument's units × level × aggregation
are part of its claim ([[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]).

*verdict_scope: INSTANCE (`A2_repeat` and `L3000_off`, the two retained lr-2e-5 runs).*

---

## §3 Control B — the temperature dose–response at fixed lr

Three arms, all lr 2e-5, same init, same seed, same bits/levers, same 600-pair eval instrument,
read at step 100. Ordered monotonically by the mean inverse CE temperature over steps 1–100:

| arm | `--float-warmup-steps` | temp @ step 100 | mean 1/temp, steps 1–100 | flips @ step 100 |
|---|---:|---:|---:|---:|
| `W1` | 100 | **1.0000** (pinned) | 1.0000 | **119,914** |
| `L3000_off` | 0 | 0.8464 | 1.0882 | 49,580 |
| `A2_repeat` | 0 | 0.4339 | 1.5635 | 27,098 |

⚠ **`W1` is a two-factor point, not a clean control.** `--float-warmup-steps 100` pins `temp = 1.0`
*and* switches the render to float (no weight quantization) for those 100 steps. It is CONSISTENT
with the temperature mechanism and it is the extreme of the axis, but it does not isolate it. The
clean pair is `A2_repeat` vs `L3000_off`.

---

## §4 The confounds on that clean pair, priced rather than asserted

`A2_repeat` and `L3000_off` differ only in `--steps`. At step 100:

- **Data order — REFUTED as a differentiator.** The retained `order` tensors are **bit-identical**
  between the runs and `cursor == 2 × step` exactly. Same batches, same sequence.
- **EMA — REFUTED as a differentiator.** Targets differ (0.99235 vs 0.99847) but both sit in the
  warm-up branch, so the effective decay is `101/110 = 0.91818` in **both**. (rg1b §F7 already
  established this dominance is structural at every `N`; the crossover is `≈1.954·N` by construction.)
- **QAT config — identical** (`--bits 4 --weight-qat-q3q4` in both).
- **Learning rate — MEASURED and priced.** Integrated lr over steps 1–100 differs by **+2.16%**. The
  lr-sweep arms give the sensitivity directly, all at the identical temperature schedule (steps=600):

| lr | flips @ step 100 |
|---:|---:|
| 2e-7 (`C0`) | 5,879 |
| 2e-6 (`A1`) | 14,960 |
| 2e-5 (`A2_repeat`) | 27,098 |
| 2e-4 (`A3`) | 63,849 |

  Local slope `β = d log(flips)/d log(lr) = 0.372` in the 2e-5→2e-4 bracket (global 0.345 over three
  decades). A +2.16% integrated-lr difference therefore predicts **+0.80%** flips. **Observed: +83.0%.**
  The learning rate accounts for **1/103rd** of the effect.

---

## §5 Control C — the damage is a rotation at constant radius

The single most discriminating control, from the Leg C checkpoint forensics
(`[macOS-CPU advisory]`, 38 tensors / 66,339 params):

| `A2_repeat` | ‖Δw‖₂ from init | flips above init |
|---|---:|---:|
| step 100 | 4.727941e-02 | 27,098 |
| step 200 | 4.702754e-02 | 12,460 |
| **change** | **−0.53%** | **2.175× better** |

The weights sit at the **same radius** from the init and get 2.175× better on argmax flips purely by
**rotating 29.97°** (cos 0.8663). `L3000_off` makes the same point monotonically across its whole CE
phase: displacement **grows 1.47×** (8.876e-02 → 1.302e-01) while flips **fall 7.6×** (49,580 → 6,510).

**Distance from the init does not price the damage. Direction does.** And the total relative
displacement at `A2@100` is only **6.354e-04** — a **0.064%** weight change buys 27,098 flips.

That is the signature of an anti-aligned objective, and it is exactly what rg1b's angle numbers
predict. Nobody had connected the two.

---

## §6 What this reframes

**The L3000 verdict's language needs one correction, and it is mine to make.** The verdict called it
"an **asymptote above init**". §1's scale-invariance says something stricter and more useful: the
alignment-weighted LR budget is **identical at every window length**, so L3000 was never a longer
experiment — it was the *same* experiment re-parameterised. "5× the window does not reach parity" is
true but understates it: **no window reaches parity**, because the quantity that would have to change
is defined as a fraction of the window. The FORMULATION-scope verdict stands and is strengthened; the
"asymptote" reading is superseded by a closed form.

**The genus.** The curriculum is a `from-scratch` schedule (soft CE first to establish global
structure, then sharpen, then margin) applied to a **fine-tune from a near-optimal init** where
`d_seg = 2.86e-4` and there is no global structure left to establish. The soft-CE phase has nothing
to gain and 99.97% of the pixels to lose. This is
[[cross-regime-constant-transfer-genus-finishing-stage]] — *re-derive latched elements at window
scope* — and the latched element here is the curriculum shape itself.

**And nobody has ever varied it.** Across **all nine** retained runs of this trainer, `ce_fraction`
is `0.5` and `softplus_fraction` is `0.85`, without exception. Three decades of learning rate have
been swept; the curriculum shape has been swept zero times.

---

## §7 The sealed ticket — NOT FIRED (MAIN owns the fire)

The cure needs **no code change**. `--ce-fraction` has no validator, and `_phase_for_step` with
`ce_fraction = 0.0` returns `softplus_margin` from step 1 (`progress = 0.0`, `0.0 < 0.0` is False).
This is a pure SCOPE change on an existing flag: **`A2_repeat` with CE removed.**

```bash
.venv/bin/python -m tac.pr130_lift.train_semantic_quantized_resumable \
  --challenge-root upstream \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
  --bits 4 --weight-qat-q3q4 \
  --steps 600 --lr 2.0e-5 --float-warmup-steps 0 \
  --ce-fraction 0.0 --softplus-fraction 0.85 \
  --eval-every 25 --checkpoint-every 25 \
  --device mps --seed 20260715 --band-objective-weight 0.0 \
  --out  /Volumes/APDataStore/pact/ddm_ce1/CE0/result.json \
  --save /Volumes/APDataStore/pact/ddm_ce1/CE0/ckpt
```

Fire through the governed launcher exactly as `L3000_off` was fired (`launch_manifest.json` schema
`detached_local_process_launch.v2`, wrapping `tools/safe_run.py --rss-mb 118784 --projected-gib 4.0`).
A **second arm** worth firing in the same batch, one flag apart — pure `expected_flip`, the
best-aligned stock objective (`cos_sign` 0.6185), for the entire run:
`--ce-fraction 0.0 --softplus-fraction 0.0`, output `…/ddm_ce1/EF0/`.

**`--eval-every 25 --checkpoint-every 25` is deliberate and is the one change from `A2_repeat`'s
cadence.** Leg C named the missing byte: no retained run has a checkpoint finer than 100 steps, so
the *shape* inside the opening excursion has never been observed. At 25 it is.

**Budget — from the MEASURED wall-clock law, not the stale figure.** `A2_repeat` measured **408 s**
for 600 steps with 6 checkpoints. With `(F = 144.3 s, r = 0.4395 s/step)` plus ~24 saves at the
~14.8 s/save term (recorded in the law as *consistent-with*, not isolated): **≈ 12 min per arm**,
≈ 25 min for both. $0, local Metal. Do **not** propagate the 33-min-per-600-step figure; it is a
measured 4.9× overprice.

### PRE-REGISTERED FALSIFIER — written before the fire

Control: `A2_repeat` at step 100 = **27,098** flips above init, same lr / init / seed / instrument.

| flips @ step 100 | reading |
|---|---|
| **< 9,000** (≥3× better) | **CONFIRMED.** The CE objective's direction is the debt. The seg axis reopens with the curriculum — not the objective's *weighting* (rg1b's band) but its *ordering* — as the live lever. |
| **9,000 – 20,000** | **PARTIAL.** CE contributes but shares the debt with a loss-form-independent fine-tune shock. Route to the init/optimizer, and re-price the band objective against the residual. |
| **> 20,000** (within ~25% of control) | **REFUTED at this formulation.** The excursion is loss-form-independent; §5's rotation reading is wrong and the routing goes to init/optimizer, not the curriculum. |

The falsifier is non-vacuous at the incumbent: the control value 27,098 sits inside the REFUTED band,
so a null result is a real possibility and would genuinely close this line.

⚠ **What a CONFIRMED result would and would not buy.** It would explain and remove the opening
excursion. It would **not**, on its own, produce descent *below* init — `best_step = 0` across nine
runs is a separate claim, and no arm has yet shown this trainer going below `2.86e-4` by any route.
Removing a +27,098 debt is a precondition for descent, not descent. Say that plainly when the row
lands rather than letting "the excursion is fixed" drift into "the seg axis is solved."

---

## §8 Retained payloads (bytes, not scalars)

| path | bytes | sha256 |
|---|---:|---|
| `/Volumes/APDataStore/pact/ddm_ce1/CE1_OPENING_EXCURSION.json` | 8,776 | `363c892df217870f9b813097828af94a4a6c1fa237b8b68be0dd704ed0ccb330` |
| `/Volumes/APDataStore/pact/ddm_ce1/ce1_legc_displacement_forensics.json` | 102,760 | `305d6d25b0ab4c3f5fbd24be8555bce8790b7e8a8eeb10f1a1857101e6259412` |
| `/Volumes/APDataStore/pact/ddm_ce1/ce1_legc_displacement_vs_flips.json` | 2,809 | `67dd91238e999e2fa80f0be8fbc61a60e54215909d5e6d71a6710232f311938d` |

Provenance pins re-hashed this unit: init ckpt `3948ccfc…` (275.7 KB) · oracle `ffdf0988…` ·
trainer `b486f416…` · L3000 verdict memo `83025c4e…` · rg1b memo `e1306299…` · wall-clock law
`ee3e061b…`.

---

## NEXT_IF_RESUMED

1. **Fire the two sealed arms in §7** (`CE0` and `EF0`), governed, ≈25 min total, $0. They are one
   flag apart and answer the pre-registered falsifier directly.
2. **Do not re-open the band objective (`rg1b` / R6) first.** It re-*weights* the gradient inside a
   phase; §1 says the dominant defect is which phase gets the LR at all — 81.2% at `cos_sign` 0.209.
   Fix the ordering, then re-price the weighting against whatever residual survives. rg1b's 45.9×
   misallocation finding is unaffected and stands.
3. **The `expected_flip` phase is starved by construction** — 0.84% of the LR budget at every window
   length. If `EF0` behaves well, the interesting knob is not "more steps" but "give the aligned
   objective the LR," which `--softplus-fraction 0.0` already does with no code change.
4. **Pin the curriculum boundaries in absolute steps** before any future length comparison (already
   the L3000 verdict's item 3 — §1 now gives the reason: fraction-of-run makes window length inert).
5. If a code change is ever wanted, the minimal one is reversing the CE temperature anneal
   (`0.08 → 1.0` instead of `1.0 → 0.08`). It is one line in the lifted oracle — but that file is a
   **reconstructive lift with pinned custody** (commit `9049e1caa5`), so touching it owes the lift
   custody protocol, not a casual edit. Prefer the zero-code-change flag route first.
