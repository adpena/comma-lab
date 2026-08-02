# ddm_rg5 — the rho is real, the mechanism is real, **the backwards-gradient conclusion is not**

**Date:** 2026-08-01 · **Axis:** `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false`
**Cost:** $0, scorer-free, training-free. No n600 slot, no launch, no dispatch.
**Pointer:** UNMOVED. This document is MEANS, not progress. No ΔS is booked below.
**Review status:** pre-registered probes (P1/P2 named before running) + own round-1 adversarial
review (§7), which added the sign-SGD arm. NOT fresh-eyes reviewed.
**Receipts:** harness `experiments/ddm_rg5_rate_gradient_sign_probe.py`; rows
`.omx/research/ddm_rg5_rows_20260801.jsonl` (159 rows: 7 meta / 120 direction / 32 permutation).

---

## §0 VERDICT, THREE CLAIMS SEPARATED

The seed bundled three things under one headline. They separate cleanly, and they do not share a
verdict.

| # | claim | verdict | how |
|---|---|---|---|
| 1 | `rho = -0.7235`, CI `[-0.943, -0.227]`, entropy surrogate vs shipped SMEVR bytes on the r1c lineage | **CONFIRMED** | re-derived at source with an independent `scipy.stats.spearmanr` over the stored rows; the committed harness re-run reproduces the stored row with **0 load-bearing mismatches** |
| 2 | mechanism: a marginal histogram is permutation-invariant, SMEVR's cost is not, so the surrogate is structurally blind to rearrangement | **CONFIRMED — and now MEASURED, not argued** | random pair-permutation moves the surrogate by `<= 4.8e-07` bits and real SMEVR bytes by **+16,062 … +18,339 B** |
| 3 | **inference:** "the rate leg **points backwards** / the term trains **AGAINST** rate on the live vehicle" | **REFUTED** | direct directional-derivative measurement on **4 fields, 4 step sizes, 3 preconditioner variants** — descent along the live gradient lowers real shipped bytes **every time, without exception**, including on both negative-`rho` checkpoints |

> **The one-line finding.** A trajectory rank correlation and a gradient sign are different objects.
> `rho` measures how two quantities co-move while the run is driven by `w_seg = 100`; it says nothing
> about where `-grad(rate)` points. Measured directly, `-grad(rate)` points **downhill in real
> shipped bytes**. The `rho < 0` is real and it is diagnostic — of a **missing** component, not a
> **reversed** one.

**⛔ The hold on `w_rate` 0.05 → 0.0768348 was NOT touched.** No config was changed, no launch fired.
The hold's *stated reason* ("that scales a backwards gradient by 1.54×") is refuted; §6 states what
would still have to be true for the hold to be correct, and that reason is *not* the one given.

### Re-derived at source vs inherited from the seed

| seed statement | status |
|---|---|
| `rho = -0.7235`, CI `[-0.943,-0.227]` | **RE-DERIVED** — independent scipy over `ddm_rsf1_rows_20260801.jsonl`, reproduces to 4 dp |
| surrogate `-1.80%` while bytes `+1.36%` (r1c ep504→640) | **RE-DERIVED** — `3.0786→3.0231` (−1.80%), `265,876→269,504` (+1.36%) |
| bc1 `-6.79%` surrogate / `+28.9%` bytes (ep109→399) | **RE-DERIVED** — `3.2665→3.0446` (−6.79%), `194,236→250,358` (+28.9%) |
| sign agreement 33–40%, "worse than a coin flip" | **RE-DERIVED** from the receipt |
| `w_seg = 100`, `w_rate` at `train_tr1…:1943`, `w_pose = 0.0` / `compute_pose=False` at `:1851-1853` | **RE-DERIVED** at file:line — all three exact |
| `token_rate_term` implements BOTH `--rate-model` modes | **RE-DERIVED** — `train_tr1…:1895-1916`, both branches live |
| `--rate-model` rung reads `UNRACED (QA86a OWED)` | **RE-DERIVED** — `spec_tr1_renderer_20260728.py:296` |
| `w_rate = 0.05` is "GENERIC-constant, **no provenance rung**" | **STALE — REFUTED at the live source.** `spec_tr1_renderer_20260728.py:287` reads `DERIVED-ESTIMATE (census T19/QA86d)`. The gd1 T19 census row said "no provenance rung found" on 07-31; a rung was written since. gc17 cites the census, not the live DSL. |
| "the **live vehicle**" / "the live lineage" | **PARTIAL.** The negative-`rho` stratum is `ddm_r1c_20260731/window_01` — the burn-4 **parent** lineage. **`v4d` is NOT in the rsf1 field set at all** (`manifest` run dirs: b4s, bc1, bp1, dw1, lv1, pa1r, r1c, tb1). The DSL rung says "burn-4 parent lineage" and is accurate; the prose framing over-reaches. |
| "SMEVR's value stream is 69.4% of bytes" | **RE-DERIVED** from the receipt (r1c stratum). Cross-checked separately: on **v4d itself** the token member `state/tokens.dr7t` is **346,478 of 360,238 B = 96.2%** of the archive, so token bytes ARE the rate axis. |

**STORES CONSULTED:** `.omx/research/ddm_gc17_from_here_gradient_not_coordinates_20260801.md` (§2, §3c,
§A1); `.omx/research/ddm_rsf1_{rows,analysis,manifest,rate_surrogate_fidelity}_20260801.*`;
`.omx/research/ddm_gd1_generic_default_census_20260731.md` (rows T4/T5/T16/T19);
`.omx/research/ddm_r7_smevr_liveness_on_v4d_20260801.json`;
`experiments/{train_tr1_partition_renderer_mlx,ddm_r7_token_coder,ddm_rsf1_rate_surrogate_fidelity}.py`;
`src/tac/witness_dsl/{spec_tr1_renderer_20260728,spec_tr1_burn2_20260731}.py`;
`tools/pb1_p5_byte_close_and_eval.py`; `git log` d619ec4ede / fe8ecbbb93.
**Deliberately NOT loaded:** the seven sibling `*waterfill*.py` modules (gc17's own named residual
scope — untouched, still open), the QA86a launch surface, and any pose-side artifact.

---

## §1 PROVENANCE OF THE RECEIPT — reproduced, with one cosmetic gap

Re-ran the **committed** harness on the checkpoint that carries the headline
(`ddm_r1c_20260731/window_01 / intra_seg_trunk_tau_ep00640.npz`) and diffed all 36 fields against
the stored row:

```
load-bearing mismatches: 0
```

Every byte column, every surrogate column, every batch-8 dispersion column matched to full float
repr. The receipt is real and reproducible.

**One gap, reported because silence about it would be the fake:** the stored rows carry three keys
the committed script does not emit — `smevr_total_bytes` (identical to `smevr_bytes` in all 70 rows),
`ckpt_sha_prefix` (**empty string in all 70 rows**), and `secs` as a `{load,smevr,brotli}` dict rather
than a float. So the rows were written by a pre-commit version of the harness. The numbers survive;
what does **not** exist is a **content hash of the checkpoints the rows were computed from** — the
`ckpt_sha_prefix` field was designed for exactly that and ships empty. Row→checkpoint custody is by
path only. That is a P1 ("one fact, one store, one key") debt, not a numbers problem.

---

## §2 P1 — THE SIGN TEST. Does `-grad` lower real shipped bytes?

**Method.** Take the live field, compute the exact gradient of the trainer's own `token_rate_term`
w.r.t. the token field (MLX autograd on the same function, never a reimplementation), normalize to
unit RMS, step `field ± alpha * (2/15) * dir` (α in **lattice units**; one quantization level is
`2/(levels-1)` wide), requantize with `quantize_tokens_np`, and encode with the **shipped** r7 SMEVR
coder. Harness: `experiments/ddm_rg5_rate_gradient_sign_probe.py`. Baselines reproduce all four rsf1
rows exactly.

### Descent lowers bytes; ascent raises them. 4 fields, 0 exceptions.

| field | `rho` in rsf1 | base B | **DESCENT** ΔB @ α=.05 / .10 / .25 / .50 | **ASCENT** ΔB @ α=.05 / .50 |
|---|---:|---:|---|---|
| `r1c` ep640 *(the headline checkpoint)* | **−0.7235** | 269,504 | **−5,336 / −7,577 / −10,441** / −6,273 | **+4,971 / +8,736** |
| `r1c` ep504 *(window start)* | −0.7235 | 265,876 | **−4,114 / −5,644 / −8,196 / −10,745** | **+3,182 / +8,815** |
| `bc1` ep399 | **−0.5382** | 250,358 | **−4,156 / −5,488 / −7,262 / −8,101** | **+5,148 / +11,644** |
| `lv1` ep399 | +1.0000 | 557,238 | **−4,567 / −6,746 / −10,276 / −13,860** | **+3,126 / +11,562** |

The sign of `rho` does not predict the sign of the byte response: `lv1` (`rho = +1.00`) and `r1c`
(`rho = −0.72`) behave **identically**. That alone separates the two objects.

### It survives both preconditioner bounds

Adam is a **sign-preserving** preconditioner (it divides each coordinate by a positive RMS estimate),
so the realized update lies in the same orthant as `-grad` but not along it. Both ends of that family
were run:

| field | raw `-grad` @.25 | **sign-SGD limit** @.25 | **batch-8 in-loop grad** @.05 / .25 | `cos(population, batch-8)` |
|---|---:|---:|---|---:|
| `r1c` ep640 | −10,441 | **−28,929** | −2,024 / −2,660 | +0.5545 |
| `bc1` ep399 | −7,262 | **−19,810** | −1,572 / −2,285 | +0.5587 |
| `lv1` ep399 | −10,276 | **−20,108** | −1,763 / −3,341 | +0.5574 |

The batch-8 gradient — **the object the optimizer actually accumulates**, not the population one —
retains `cos ≈ +0.556` with the population gradient and lowers bytes on every field at every step.
The in-loop estimator is **noisy and correctly signed**, not reversed.

### Control

A random unit-RMS direction is not a byte-reducer. On the unmasked field (`lv1`, `keep_frac = 1.0`,
the clean control) random gives **+81 B** at α=.05 where the entropy gradient gives **−4,567 B**. On
the masked fields (`keep_frac = 0.5`) random gives **+126k … +128k B** — that number is dominated by
noise destroying the 50% constant zero-cells, so the masked random arm is a *weak* control and only
the `lv1` figure is quoted as one. Identity/zero-step reproduces the baseline **exactly** (negative
control passes on all four fields).

---

## §3 P2 — THE PERMUTATION TEST. The mechanism is real; here is its SIZE.

Permuting the pair axis leaves the pooled marginal histogram's multiset identical by construction,
so it is the exact discriminator for gc17's mechanism claim, and it needs no correlation.

| field | surrogate Δ under random permutation | SMEVR bytes Δ | reverse-order Δ | identity |
|---|---:|---:|---:|---:|
| `r1c` ep640 | `<= 4.8e-07` bits | **+16,062 … +16,181** | −32 | 0 |
| `r1c` ep504 | `0.0e+00` bits | **+17,223 … +17,362** | −32 | 0 |
| `bc1` ep399 | `<= 2.4e-07` bits | **+18,188 … +18,339** | −42 | 0 |
| `lv1` ep399 | `<= 4.8e-07` bits | **+18,031 … +18,154** | −23 | 0 |

**The mechanism is CONFIRMED and quantified: there is a ~16–18 kB byte subspace reachable by pure
rearrangement that the live surrogate provably cannot see.** (~16.1 kB = **0.0107 S** at the exact
rate exchange.) Time-reversal is nearly free (−23 … −42 B), so SMEVR's context is close to
reversal-symmetric; the cost lives in *local* temporal coherence, which random permutation destroys.

### And this is what actually explains the `rho`

| field | permutation-blind subspace | that field's trajectory byte range | ratio |
|---|---:|---:|---:|
| **`r1c` ep504→640** | ~16.1–17.3 kB | **4,353 B** | **3.7–4.0×** |
| `bc1` ep9→399 | ~18.3 kB | 164,075 B | 0.11× |
| `lv1` ep9→399 | ~18.1 kB | 402,939 B | 0.04× |

On the headline stratum the **entire trajectory byte variation is 3.7–4.0× smaller than the blind
subspace**. The observed anti-correlation lives *inside* the surrogate's provable blind spot. An
anti-correlation measured inside a blind spot is evidence of a **missing** component, and is
consistent with a gradient whose visible component points correctly — which §2 shows it does.

### Three further reasons that stratum should not have carried the headline

Read off the same receipt, all in its own printed output:

1. **It is the lowest-SNR stratum by 7.7–35×.** rsf1's own noise-floor block: `B_traj_r1c` entropy
   between-field spread 0.0570 bits vs batch-8 std 0.0160 → **SNR 3.56×**; the other three strata are
   27.4× / 77.6× / 123.0×.
2. **It has the smallest byte dynamic range by 116×** — 1.64% of min, against 190% / 261% / 6206%.
3. **The ORACLE flips negative there too.** `hard mode-occupancy` — the non-differentiable
   ground-truth diagnostic — is **−0.6941** on `B_traj_r1c` while it is `+0.87` pooled and `+0.97`/`+1.00`
   elsewhere. When the oracle and the surrogate *both* anti-correlate on one stratum, the stratum is
   telling you about itself, not about the surrogate.

**The receipt's other strata were not cited in gc17 §3c:** pooled `rho = +0.2832` (n=70),
`A_crossconfig +0.3424`, `D_traj_lv1 **+1.0000** with 100% sign agreement`, and `C_traj_bc1`'s CI
`[-0.988, **+0.110**]` **includes zero**. The DSL rung at `spec_tr1_renderer_20260728.py:313` *does*
carry the regime scope ("entropy tracks at rho=+1.00 in the filling regime") — the DSL leg is more
honest than the memo prose it was written from.

---

## §4 THE EXCHANGE-RATE PREMISE BEHIND 0.0768348 — measured, and it holds

`derive_w_rate_exchange_rate` (`spec_tr1_burn2_20260731.py:62-78`) derives
`w_rate = (25/37,545,489) · n/8` from the premise **"reducing the surrogate's mean by 1 bit/token
saves `n/8 = 115,392` bytes."** The rung itself flags that premise as approximate. rsf1 regressed it
**along trajectories** and got **−45,228 B/bit** on `B_traj_r1c` — negative, i.e. apparently refuting
it. That regression inherits the same confound as `rho`.

Measured as an actual **directional derivative** (α ≤ 0.25, both signs, n=6 per field):

| field | measured B/bit (min / mean / max) | vs assumed 115,392 |
|---|---|---:|
| `r1c` ep640 | 94,491 / **98,394** / 102,623 | 0.85× |
| `r1c` ep504 | 95,477 / **111,555** / 163,656 | 0.97× |
| `bc1` ep399 | 87,347 / **97,770** / 112,572 | 0.85× |
| `lv1` ep399 | 139,165 / **160,554** / 180,575 | 1.39× |

**Mean over the four fields: 117,068 B/bit — within 1.5% of the assumed 115,392, sign correct,
spread 0.85–1.39×.** The premise the 0.0768348 derivation rests on is **measured-supported**, not
refuted. The trajectory regression's negative slope is an artifact of the same confound as the `rho`.

---

## §5 WHAT THE MEASUREMENT ACTUALLY RECOMMENDS — the race, on better evidence

`--rate-model=smevr_surrogate` is genuinely built (`train_tr1…:1901-1916`) and genuinely unraced
(rung `UNRACED (QA86a OWED)`, race programs already written at
`spec_tr1_burn2_20260731.py:96-111`). Order #3's premise holds: the race is a config flip.

But the case for it does **not** need the contested `rho`. At **identical RMS step budget**:

| field | entropy best ΔB | `smevr_surrogate` best ΔB | ratio | `cos(entropy, smevr_surrogate)` |
|---|---:|---:|---:|---:|
| `r1c` ep640 | −10,441 | **−16,116** | 1.54× | **−0.0664** |
| `r1c` ep504 | −10,745 | **−23,580** | 2.19× | **+0.0204** |
| `bc1` ep399 | −8,101 | **−21,928** | 2.71× | **−0.0554** |
| `lv1` ep399 | −13,860 | **−49,566** | 3.58× | **−0.0092** |

Two things fall out, and the second is the more useful:

1. `smevr_surrogate` is a **1.5–3.6× stronger byte-descent direction** on every field. Where entropy's
   curve turns back up (r1c ep640, α=0.5), `smevr_surrogate`'s keeps descending. The race is
   warranted.
2. **`cos(entropy, smevr_surrogate) ≈ 0` on all four fields (−0.066 … +0.020).** They are **very
   nearly ORTHOGONAL, not opposed.** They descend the same cost through *different* coordinates.
   A two-way A/B therefore tests the wrong hypothesis space: the indicated third arm is the **SUM**
   (`entropy + smevr_surrogate`), which the near-orthogonality predicts is close to additive. That arm
   does not exist in `qa86_rate_surrogate_race_programs`, which builds exactly two.

**No ΔS is booked for any of this.** gc17's own backcast measured this format's bookings at ~100×
optimistic, and these are **static-field byte responses at a frozen operating point**, not training
outcomes. They are evidence about the *sign and relative strength of a direction*, nothing more.

---

## §6 THE HOLD — the stated reason is void; a different reason may survive

**Not touched. No config edited, no launch fired.** Reporting the assessment only.

The order's reason — *"if the gradient's sign is wrong, that scales a backwards gradient by 1.54×"* —
is **refuted**: the gradient's sign is right on 4/4 fields, 3/3 preconditioner variants. And §4 shows
the derivation's own premise is measured-supported. So the two stated grounds for the hold are gone.

**What is still genuinely unmeasured, and is the honest reason a hold could remain correct:** every
number in this memo is a **byte** number. A byte reduction bought by moving the token field is not
free — it is bought in `d_seg`, and the admission rule is `W = 1.2731 B/flip`. **This probe never ran
the scorer.** At α=0.25 on `r1c` ep640 the entropy step buys −10,441 B (= −0.006952 S on the rate
term); if that same step costs more than `10,441 / 1.2731 = 8,201` pixel flips, it is a net **loss**.
Nothing here bounds that. Raising `w_rate` 1.54× moves along a direction now known to be correctly
signed *in bytes* and **unknown in the joint objective**.

That is exactly what QA86a measures natively — "matched budget, SMEVR byte ledger on both, falsifier
at matched `d_seg`." **The owed measurement IS the race.** It is a heavy launch and stays operator-GO;
I did not fire it.

---

## §7 MY OWN ROUND-1 ADVERSARIAL REVIEW (attacks I ran on myself)

| attack | resolution |
|---|---|
| "A field-space gradient is not the parameter-space update Adam applies." | The token field **is** the parameter (`tokens_base + tokens_delta`, both `npz` params). Adam preserves each coordinate's sign, so its update stays in the `-grad` orthant. **Ran the sign-SGD limit** (the far end of that family) — descent still lowers bytes, more strongly (§2). Attack closed by measurement, not argument. |
| "You used the population gradient; the trainer uses batch-8." | Ran the batch-8 gradient too (32 draws averaged). `cos = +0.556`, descends bytes on all fields. Included in §2. |
| "`smevr_bytes` is not shipped archive bytes." | The byte-close path `tools/pb1_p5_byte_close_and_eval.py:239` and the trainer's own ledger `train_tr1…:695-712` both call `encode_token_codes(codes, levels=16, codec="smevr")` on the same full `(P,gh,gw,c)` array. On v4d the token member is **96.2%** of the archive. Fair. |
| "The EMA basis is not the live-parameter basis the gradient acts on." | **Conceded, unresolved.** `load_field` prefers `ema::`; I perturb the shipped (EMA) basis. Sign conclusions on a 4/4 unanimous result are unlikely to flip, but this is a labeled residual, not a closed one. |
| "α = 0.05 lattice units — is that local?" | 0.05 lattice-bins RMS displacement; most tokens do not cross a bin. Bytes still move −4.1k … −5.3k. Genuinely local. |
| "Do your baselines match rsf1's?" | All four: 269,504 / 265,876 / 250,358 / 557,238 B and 3.023075 / 3.078577 / 3.044621 / 3.5696 bits — exact. Two independent harnesses agree. |
| "Denominator honesty." | Field set: **4 fields** (2 of the 3 negative-`rho` checkpoints named in gc17 plus the positive-`rho` control plus the r1c window start), drawn from a receipt of **70**. Strata covered: 3 of 4 (`A_crossconfig` **not** probed — a named, uncleared residual). Permutation arms: 6 random + reverse + identity per field. |
| "Is the autograd result actually the gradient?" | **Checked, not assumed.** Central finite differences on the 6 largest-magnitude kept coordinates of a synthetic field through the same code path: max relative error **1.9e-03** (`entropy`) and **3.9e-03** (`smevr_surrogate`) — fp32 finite-difference noise. Non-kept cells carry **exactly zero** gradient (no mask leak). |
| "Would your test pass if the code were broken?" | No: identity permutation and α=0 must return the byte count **exactly**, and do (0 delta, 4/4). A broken encode or a stale field would break that first. |

**Verdict scope.** REFUTED-item #3 is scoped **FORMULATION** — the "backwards gradient" reading of
`rho` is refuted on the TR1 `entropy` rate term across 4 fields and 3 strata; it is not a claim about
any other surrogate or vehicle. CONFIRMED items #1 and #2 are **MEASURED** at the strata named.
`A_crossconfig` and `v4d` itself are **UNMEASURED** by this probe.

---

## §8 RESIDUALS — named, not cleared

1. **`A_crossconfig` (n=22) never probed** for the sign test. Its `rho` is `+0.3424`; nothing here
   contradicts it, but it is untested.
2. **`v4d` — our actual frontier vehicle — is in neither rsf1's field set nor mine.** Every rate-leg
   statement in circulation is about the burn-4 **parent** lineage.
3. **The `d_seg` price of the byte reduction is unmeasured** (§6). This is the binding gap.
4. **Row→checkpoint custody is path-only** — `ckpt_sha_prefix` ships empty in all 70 rsf1 rows (§1).
5. **The `entropy + smevr_surrogate` sum arm does not exist** in `qa86_rate_surrogate_race_programs`
   (§5). Building it is a DSL edit, not a launch.
6. **gc17's seven unopened `*waterfill*.py` siblings** remain unopened here too.
