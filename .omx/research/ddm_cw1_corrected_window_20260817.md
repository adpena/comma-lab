---
arm: ddm_cw1_corrected_window
title: "all three findings REPRODUCE EXACTLY from primary artifacts (13.6069x / 92.651%; 81.1497-81.1997% scale-invariant; F=-17.412 refuted, F=+122.290 accepted, 4.849x overprice) -- and re-deriving them surfaced six things the memos did not have, the largest being that the FRONTIER VEHICLE HAS NO SEG TERM IN ITS LOSS AT ALL, and the second that EF6000's 'interior best = a floor' is confounded with an anneal that had 0.64% of its LR budget left"
utc: 2026-08-17
charter: "ddm_cw1 -- re-derive three landed findings, compose them into ONE corrected training window, seal a ticket. $0, no launches."
axis: "[macOS-MPS training-signal] re-analysis of already-retained payloads + [macOS-CPU advisory] source inspection. NO training ran. NEVER a score."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-finding; the closed forms are DERIVED and exact, the arm numbers are INSTANCE on the named retained runs"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_cw1 — the corrected window

**STORES CONSULTED (read at source, never from a summary):**
`.omx/research/ddm_ce1_ce_opening_excursion_mechanism_20260817.md` ·
`.omx/research/ddm_ce1_allocation_ladder_verdict_20260817.md` ·
`.omx/research/ddm_wallclock_eval_cadence_refit_20260817.md` ·
`.omx/research/ddm_wallclock_prefix_bias_law_20260817.md` ·
`.omx/research/ddm_ef3000_first_descent_verdict_20260817.md` ·
`.omx/research/ddm_ef6000_double_window_verdict_20260817.md` ·
`.omx/research/ddm_l3000_no_descent_verdict_20260817.md` ·
`.omx/research/ddm_ws4_budget_allocation_audit_20260817.md` ·
`.omx/research/ddm_q3q4_owed_control_verdict_20260817.md` ·
`.omx/research/ddm_drain_vehicle_split_and_lever2_payoff_20260817.md` ·
`.omx/research/ddm_frd077_lever_verdict_and_zero_row_nan_20260817.md` ·
`.omx/research/ddm_rar1_fresh_eyes_review_findings_20260817.md` ·
`.omx/research/ddm_rg1b_band_objective_build_20260816.md` §6.2 ·
`src/tac/pr130_lift/lifted/semantic_renderer_oracle.py` ·
`src/tac/pr130_lift/train_semantic_quantized_resumable.py` ·
`tools/train_ddm_cl1_hpac_capacity.py` + `tools/train_ddm_cl1_hpac_capacity_mps.py` ·
retained payloads `/Volumes/APDataStore/pact/{ddm_jr1,ddm_lr1,ddm_ce1}/*` and the eight launcher
`detached_local_process_done.v2` receipts in `.omx/tmp/codex_runs/` (all READ-ONLY).

---

## ANSWER FIRST

**All three findings reproduce exactly.** Not approximately — to every digit the memos quoted.

| # | claim | memo | ddm_cw1 re-derivation | verdict |
|---|---|---|---|---|
| 1 | seg wall is configuration | 13.6× / 8,018 flips / 92.7% | **13.6069× / 8,018 / 92.651%** | REPRODUCED |
| 2 | LR-budget split, scale-invariant | 81.15 / 81.19 / 81.20% | **81.1497 / 81.1905 / 81.1997%** | REPRODUCED |
| 2 | aligned budget | 26.88 / 26.87 / 26.87% | **26.8841 / 26.8707 / 26.8677%** | REPRODUCED |
| 3 | save model refuted (F<0) | F = −17.41 s | **F = −17.41223 s** | REPRODUCED |
| 3 | eval model accepted | F=+122.29, r=0.22267, e=25.400 | **+122.28999 / 0.222669 / 25.40040** | REPRODUCED |
| 3 | window overprice | "a measured 4.9× overprice" | **4.849×** | REPRODUCED |
| 3 | eval share of the apparent rate | ~53% | **52.0%** (8-point refit) | REPRODUCED |

**Nothing failed to reproduce.** That is the honest headline and it is worth stating plainly: this repo
has had six stale-headline incidents, and the correct outcome of a re-derivation is usually *some*
correction. Here the arithmetic held. The corrections below are not to the three claims — they are to
things the three memos did not check.

**Six findings the re-derivation surfaced, ranked by blast radius:**

1. **The frontier vehicle has NO seg term in its loss at all.** `tools/train_ddm_cl1_hpac_capacity.py:1234`
   is `task_loss = F.cross_entropy(token_logits, tokens)` plus a rate term, and `:1265` selects `best`
   on `estimated_joint_bytes`. SegNet never enters the loop. The vehicle carrying **0.029611 S of seg
   debt** optimises token density and picks its checkpoint on bytes. On the seg axis it is not
   *misaligned* — it is **absent**. VERIFIED_VIA_SOURCE_INSPECTION.
2. **EF6000's "interior best ⇒ this regime has a floor" is confounded with the anneal.** At its best
   step (5,200) the cosine had **0.64% of its integrated LR budget left** and lr was 5.28% of peak.
   EF3000's best sat at the cap with **0.00%** left. "Approached a floor" and "ran out of schedule"
   are not separable on this evidence. This is the L3000 fraction-confound one level down.
3. **The (lr × curriculum) grid was never crossed.** Three decades of lr were swept **only** under the
   misaligned curriculum; the aligned curriculum is measured at **exactly one** lr. `--lr 2e-5` is a
   BORROWED CONSTANT for this objective.
4. **The `tau` anneal inside `expected_flip` is itself a fraction-of-run constant, now stretched 6.7×.**
   At `softplus_fraction = 0.0` it sweeps 0.15→0.05 across the whole run instead of the last 15%. The
   three EF arms are matched in progress-space (good), but "more steps" silently means "slower tau per
   step", so EF6000's 1.205× sublinear return is **not separable** from it. No flag. Never swept.
5. **The EMA target is INERT on every window measured.** The warm-up ramp `(1+t)/(10+t)` reaches the
   LawRef-derived target at t\* = 1,167 / 5,857 / 11,720 for the 600 / 3,000 / 6,000-step arms — always
   beyond the run. `ema_decay_run_geometry_v1` is correctly consulted (`fallback_used=False`) and its
   output never binds. **A composed window may not claim its EMA basis was tuned.**
6. **Eval cadence is a selection-bias knob, not only a cost knob.** Evaluation is trajectory-NEUTRAL
   (source-verified below), so the **endpoint is cadence-invariant**. But `best` is a min over the eval
   samples: EF6000's detrended tail scatter is sd 465 flips over 14 evals, giving ≈790 flips of pure
   selection bias — which is most of the 135-flip gap between its best (−2,755) and its endpoint (−2,620).

**Pointer UNMOVED.** No training ran, no archive was built, no score was produced. This unit is MEANS.

---

## §1 Claim 1 re-derived — 13.6069×, 92.651%

Read from each arm's own `result.json`, never from a memo. All three share the init byte-for-byte
(`semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`, `init_quantized_exact_seg =
2.861616346571e-04` identical to 12 digits = **33,757.0 flips** at the 117,964,800-pixel denominator),
the same seed 20260715, the same lr 2e-5, the same 600 steps, `--weight-qat-q3q4` ON in all three.

| arm | `--ce-fraction` | `--softplus-fraction` | endpoint flips vs init |
|---|---:|---:|---:|
| `A2_repeat` | 0.50 | 0.85 | **+8,654** |
| `CE0` | 0.00 | 0.85 | +4,852 |
| `EF0` | 0.00 | 0.00 | **+636** |

`8654 / 636 = 13.6069×`; `8654 − 636 = 8,018 = 92.651%`. Monotone `control > CE0 > EF0` at **every**
one of the six shared checkpoints (100/200/300/400/500/600) — I checked each rather than the endpoint
alone, because a monotonicity claim is exactly the kind that survives on one lucky endpoint.

**Extended by me, since it was cheap:** the same shape at longer windows crosses BELOW the init where
ten prior runs could not — `EF3000` endpoint **−2,286**, `EF6000` endpoint **−2,620** (best −2,755 at
step 5,200). And the owed `q3q4` control has since run: `Q3Q4OFF` endpoint −1,559, best −1,723.

*verdict_scope: INSTANCE (the named retained runs, one seed, the trainer's advisory
`quantized_exact_seg`).*

---

## §2 Claim 2 re-derived — a closed form, not a fit

The derivation needs only two source facts, both read at source:

* `semantic_renderer_oracle.py::curriculum_loss` — `progress = step / max(total_steps-1, 1)`; `ce` while
  `progress < ce_fraction`, `softplus_margin` while `progress < softplus_fraction`, else `expected_flip`.
* `train_semantic_quantized_resumable.py:1066` — `CosineAnnealingLR(optimizer, T_max=args.steps,
  eta_min=args.lr * 0.01)`, stepped once per iteration.

Both the phase selector and the schedule are parameterised by **fraction of the run**, so each phase's
share of the integrated learning rate is a function of the fractions alone:

| T | `ce` | `softplus_margin` | `expected_flip` | aligned budget |
|---:|---:|---:|---:|---:|
| 600 | 81.1497% | 18.0075% | 0.8427% | 26.8841% |
| 3,000 | 81.1905% | 17.9728% | 0.8367% | 26.8707% |
| 30,000 | 81.1997% | 17.9650% | 0.8353% | 26.8677% |

against `cos(sign g)` at the shared init of **0.2087 / 0.5235 / 0.6185** — re-read at source from
`ddm_rg1b_band_objective_build_20260816.md` lines 481–483, the real gradient path, not a summary.

Under `--ce-fraction 0.0 --softplus-fraction 0.0` the aligned budget is **61.85%** at every window
length — a **2.30×** improvement, and the whole of claim 1's effect.

**Why no window could ever have fixed it:** the quantity that would have to change is *defined* as a
fraction of the run. Lengthening the run re-parameterises it and leaves it invariant. That is a
derivation, not an observation, which is why it holds at 30,000 steps as well as at 600.

*verdict_scope: DERIVED and exact, given the two source facts above.*

---

## §3 Claim 3 re-derived — and extended from 3 points to 8

All elapsed figures come from ONE instrument: the launcher's `detached_local_process_done.v2` receipt
`elapsed_s`. I checked that at source before fitting, because mixing a launcher-measured elapsed with a
trainer-measured one is this campaign's recurring defect.

| run | steps | evals | saves | curriculum | elapsed_s |
|---|---:|---:|---:|---|---:|
| `A2_repeat` | 600 | 6 | 7 | mixed | 408.294 |
| `L3000_off` | 3,000 | 30 | 13 | mixed | 1552.309 |
| `CE0` | 600 | 24 | 25 | softplus+ef | 865.501 |
| `EF0` | 600 | 24 | 24 | expected_flip | 842.000 |
| `EF3000` | 3,000 | 30 | 12 | expected_flip | 1445.570 |
| `EF6000` | 6,000 | 30 | 12 | expected_flip | 2042.649 |
| `FRD077` | 3,000 | 30 | 12 | expected_flip | 1451.780 |
| `Q3Q4OFF` | 3,000 | 30 | 12 | expected_flip | 1441.080 |

The 3-point models reproduce to the digit: **A** (save-dominated) `F = −17.41223 s` — unphysical,
REFUTED; **B** (eval-dominated) `F = +122.28999, r = 0.222669, e = 25.40040` — accepted. The stale
33-min-per-600-steps figure against `A2_repeat`'s measured 408.294 s is a **4.849×** overprice.

**The extension the memo's own NEXT_IF_RESUMED asked for is already payable.** Five more receipts exist
and `L3000_off` / `EF0` break the eval-save collinearity, so a least-squares fit over all eight points
separates the terms the 3-point fit could not:

| model | F (s) | r (s/step) | e (s/eval) | s (s/save) | max resid | rms |
|---|---:|---:|---:|---:|---:|---:|
| B, 8 pts, one `r` | +140.40 | 0.19245 | 25.076 | — | 82.28 s | 33.30 s |
| **C, per-curriculum `r`** | **+126.20** | **ef 0.19429 / mixed 0.22812** | **24.758** | — | **10.72 s** | **6.35 s** |
| D, + saves | +114.32 | ef 0.19886 / mixed 0.23382 | 23.930 | +1.442 | 5.69 s | 2.70 s |

Model **C** reproduces every retained run within **0.95%**; the 3-point law extrapolated to `EF6000`
at **−8.0%**. The per-curriculum split confirms the `ef6000` memo's residual as mechanism: the
`expected_flip` phase is **14.8% cheaper per step** than the mixed curriculum the law averaged over.
Model D additionally recovers a small, physical, positive save cost (**+1.44 s**), which is the sign the
3-point save model could not produce.

**Eval share of the apparent mixed rate at cadence 100:** `(24.758/100) / (0.22812 + 24.758/100) =
52.0%` — the memo's "roughly 53%", confirmed on 2.7× the data.

**Recommended quotation from here on:** `F = 126.198 s`, `r_expected_flip = 0.19429 s/step`,
`r_mixed = 0.22812 s/step`, `e = 24.758 s/eval`, **with the eval cadence stated**. Model D's `s` is
real but small; folding it in changes a 17-minute prediction by ~20 s.

*verdict_scope: INSTANCE on this trainer, this host, `child_owned` resource mode (no thread caps).*

---

## §4 The load-bearing check nobody had run: is the eval cadence a confound?

The allocation ladder compares `A2_repeat` at `--eval-every 100` against `CE0`/`EF0` at `--eval-every 25`.
If evaluation perturbed the trained trajectory, the ladder's control would be confounded and claim 1
would be worth nothing. I traced it rather than assuming:

* `_evaluate_semantic_pairs` and `_evaluate_rgb_pairs` are `@torch.no_grad()`.
* `ema_eval_scope` (`:248-261`) snapshots `model.state_dict()` with `.detach().clone()`, applies the
  shadow, and in `finally` calls `model.load_state_dict(original)` and `model.train(was_training)` —
  an **exact** restore, and the pattern CLAUDE.md's EMA rule requires.
* The train loop re-asserts `model.train()` at the top of every step (`:1213`) regardless.
* The eval iterates an explicit `pair_ids` tensor and draws **nothing** from the run's
  `torch.Generator`; the generator is consumed only by `torch.randperm` on batch exhaustion.

**Conclusion: cadence is trajectory-neutral. The ladder is not confounded, and the ENDPOINT is
cadence-invariant** (the endpoint eval always fires via `step == args.steps`). What *is*
cadence-dependent is `best`, and §5 prices that.

---

## §5 The five corrections, with their arithmetic

### 5.1 The frontier vehicle has no seg term (the biggest one)

`tools/train_ddm_cl1_hpac_capacity_mps.py` delegates by pinned content hash to
`tools/train_ddm_cl1_hpac_capacity.py`. Its inner loop:

```python
target = tokens[idx]
logits = model(target, idx, previous[idx])
task_loss = F.cross_entropy(logits, target)                       # :1234
rate_loss = args.rate_lambda * math.log(2) * variable_weight_bits(model, deployed=False) / pixels
loss = task_loss + rate_loss
```

and its checkpoint selector: `if best is None or metrics["estimated_joint_bytes"] < best[...]` (`:1265`),
over metrics `{bpp, top1_error, estimated_token_bytes, estimated_model_bytes, estimated_joint_bytes}`.
**SegNet appears nowhere in the training loop.** The frontier's d_seg (0.029611 S = 34,930 flips) is a
downstream by-product of an objective that is token density plus rate, selected on bytes.

It also carries the same two structural constants the `ce1` line found poisonous: a fraction-of-run stage
boundary (`qat_start = max(1, floor(epochs*(1-qat_fraction))+1)`, `:1041`) and
`CosineAnnealingLR(T_max=args.epochs, eta_min=args.lr*0.02)` (`:1037`).

**What this does and does not license.** It does NOT transfer the number 0.2087 — that was measured for
CE-over-SegNet-logits on a different architecture, and a borrowed number is a hypothesis here. It DOES
settle the vehicle question completely: `--ce-fraction` / `--softplus-fraction` **do not exist** on the
frontier trainer and there is no aligned-objective phase to allocate to, so the composed window cannot
run there. And it names the cheap probe: `rg1b`'s `cos(sign g)` instrument, pointed at the frontier
trainer's own gradient, would say whether its objective points where its score does. That probe is
built and has never been aimed there.

*verdict_scope: VERIFIED_VIA_SOURCE_INSPECTION (a code fact about two files). The consequence for the
frontier's d_seg is INFERRED and un-measured.*

### 5.2 EF6000's interior best is anneal-confounded

| arm | best step | lr there | % of peak | LR budget remaining after it |
|---|---:|---:|---:|---:|
| `EF3000` | 3,000 (cap) | 2.000e-07 | 1.00% | **0.00%** |
| `EF6000` | 5,200 (interior) | 1.056e-06 | 5.28% | **0.64%** |

The `ef6000` memo's claim 3 reads: *"the first evidence this regime has a floor it approaches rather
than a cap it is cut off at."* A run that stops improving with 0.64% of its learning rate left has not
demonstrated a floor. It has demonstrated that it stopped when the schedule stopped. Correcting my
sibling's headline, not its measurement: the endpoint numbers stand; the mechanism claim does not.

### 5.3 The (lr × curriculum) grid

| | lr 2e-7 | 2e-6 | 2e-5 | 2e-4 |
|---|:-:|:-:|:-:|:-:|
| stock 0.50 / 0.85 | X | X | X | X |
| ce0 0.00 / 0.85 | · | · | X | · |
| **aligned 0.00 / 0.00** | · | · | **X** | · |

Under the stock curriculum the measured response was **damage rising as lr^0.372**. That is the response
of a *misaligned* objective to more learning rate. Under an aligned objective the sign is untested, and
`2e-5` was inherited from the arm the stock sweep happened to centre on. **This is the cell the composed
window fills.**

### 5.4 `tau` is the next inherited fraction-of-run constant

`tail = (progress − softplus_fraction) / max(1 − softplus_fraction, 1e-6)`; `tau = 0.15 − 0.10·tail`.
Hardcoded in the lifted oracle; no flag. Under the stock shape it swept its whole 0.15→0.05 range across
the last 15% of the run; at `softplus_fraction = 0.0` it sweeps the same range across 100%. The three EF
arms remain matched in progress-space, so their comparison is clean — but the *interpretation* of
"longer window" is not, because a longer window is also a slower tau anneal per step.

### 5.5 The EMA target is inert

Effective decay is `min(target, (1+t)/(10+t))`; the ramp reaches the target at `t* = (10·target − 1)/(1 − target)`.

| arm | steps | LawRef target | t\* | binds? |
|---|---:|---:|---:|---|
| `A2_repeat` / `CE0` / `EF0` | 600 | 0.992354096 | 1,167 | **no** |
| `L3000_off` / `EF3000` | 3,000 | 0.998466121 | 5,857 | **no** |
| `EF6000` | 6,000 | 0.999232766 | 11,720 | **no** |

`ema_decay_run_geometry_v1` is consulted correctly (`fallback_used = False`, `governing_policy =
warmup_ramp`) and its output never binds. This is not a defect in the law — it is `rg1b` §F7's measured
structural dominance of the ramp, restated at these window lengths. It is a **claim discipline**: no
composed window may say its EMA basis was tuned.

### 5.6 Cadence is a bias knob

`EF6000`'s tail from step 3,400 (n = 14 evals): linear trend **−1,063 flips per 1,000 steps** (real
descent), residual sd **465 flips**. The best sits **−1.86 sd** below its own trend line; the expected
minimum of 14 standard normals is −1.70 sd ≈ **−790 flips of pure selection**. The endpoint's residual is
**+0.26 sd**.

So: **report the endpoint.** Quote `best` only with its eval count attached — otherwise a finer cadence
manufactures a better number out of the same run.

---

## §6 The composed window

**Vehicle: the `pr130_lift` semantic renderer** (38 tensors, 66,339 params, `blocks.{0..3}` + FiLM).
Declared, with its limit stated: this is **not** `hv1` (37 tensors, 39,375 params, no FiLM) — a different
architecture, not a different checkpoint. It is the vehicle I configure because §5.1 proves it is the
**only** vehicle on which the measured lever exists: the frontier trainer has no seg objective to
re-allocate. **This window does not move the exact score.** It measures the depth of the only
seg-descent regime anyone has measured, for $0, which is the precondition for deciding whether to pay
for the transfer measurement — and the transfer is where the S actually lives.

**Objective allocation.** `--ce-fraction 0.0 --softplus-fraction 0.0`: 100% `expected_flip`, aligned
budget 61.85% vs the stock 26.87%. MEASURED best of three. **Not a proven optimum** — `cos(sign g) =
0.6185` is the best of three measured objectives, not a ceiling, and nothing here exhausts that axis.

**LR schedule.** `CosineAnnealingLR(T_max=steps, eta_min=0.01·lr)` — unchanged, because there is no
zero-code lever that decouples `T_max` from `--steps`. `--lr` is therefore the schedule lever: since
`eta_min = 0.01·lr`, raising it lifts the tail as well as the peak, which is exactly what §5.2 says both
prior windows needed. **`--lr` is the swept axis of this window**, and it is carried as a typed
class-4 waiver in the DSL rather than as a number, because its aligned-objective optimum is unmeasured.

**Eval cadence.** `--eval-every 250` (13 evals at 3,000 steps): 31% of wall-clock at the measured
24.758 s/eval, against 43% at cadence 100 and 70.6% at cadence 25. Resolves the tail at 250-step
granularity, which is what "still descending at the cap?" needs, without inflating the best-of-N band.

**Checkpointing (P0).** `--checkpoint-every 200` (15 saves, ~1.6 MB each, ~1.4 s each). Deliberately
**unequal to `--eval-every`** — equal cadences are what made eval and save collinear and produced a
wall-clock law implying a negative fixed cost. Saves are atomic `tmp+rename`, stage-encoded
(`ckpt.stage-expected_flip.step003000.full_state.pt`), carry the EMA shadow, and resume restores model,
optimizer, scheduler, EMA and the batch generator state — verified against the retained arms.

**Stopping rule.** Fixed 3,000 steps, and the reported metric is the **endpoint**, not the best. Rationale
in §5.6. `best_step` is still recorded; it is read as *diagnostic of where the schedule stopped helping*,
never quoted as the result.

**EMA basis.** `ema_decay_run_geometry_v1` at the trainer default seed fraction 0.01. Declared **inert**
per §5.5 — the warm-up ramp governs the whole 3,000-step window. Stated so no downstream row credits it.

### The ladder

| rung | steps | lr | integrated LR | vs EF3000 | status |
|---|---:|---:|---:|---:|---|
| 1 | 3,000 | 2.0e-5 | 0.03031 | 1× | **`EF3000`, ALREADY MEASURED, free** (endpoint −2,286) |
| 2 | 3,000 | 6.0e-5 | 0.09093 | 3× | **`CW1-LR6E5`, sealed** |
| 3 | 3,000 | 2.0e-4 | 0.30310 | 10× | **`CW1-LR2E4`, sealed, CONDITIONAL** |

Integrated LR over a cosine run is exactly `steps × 0.505 × lr`, so lr and steps buy the same currency —
and at fixed steps the tau schedule is held identical to rung 1, making lr the only variable.

**Fire order and gate: rung 2 first.** If rung 2 lands in the NULL or REFUTED band, rung 3 is **not**
fired on the depth question (its answer is already implied and the 17 minutes are better spent on the
objective axis). This is a gate, not a fan-out.

### Projected wall-clock, with (F, r, e) and the cadence stated

`F = 126.198 s` · `r_expected_flip = 0.19429 s/step` · `e = 24.758 s/eval` · `--eval-every 250`
⇒ evals = 1 + 3000/250 = 13 ⇒ **1,030.9 s = 17.2 min per rung**; both rungs **34.4 min**, $0 local Metal.
(12% fixed / 57% training / 31% evaluation.) Under the superseded law the same pair would have been
priced at 49 min; under the pre-`aa3` figure, 330 min — **9.6× the corrected price.**

### Projected Δflips and its band — an interval, and an honest power limit

The only two aligned points available are EF3000 (−2,286 at 1×) and EF6000 (−2,620 at 2× integrated LR),
which fit `Δ ∝ (integrated LR)^0.197`. **I do not adopt that as the prediction.** It is n = 2, and this
campaign struck a linear-tail extrapolation today for exactly this reason; a power law on two points
deserves the same treatment. For the record it would give **−3,138** at rung 2 and **−3,596** at rung 3.

The pre-registered prediction is an **ordering plus a band**, using the measured two-run comparison band
σ = 605 × √2 = **856 flips** (605 = the A/A difference at step 600 between `A2` and `A2_repeat`, same
config, same seed — MPS nondeterminism alone).

**Stated power limit, before the fire:** an LR-budget-matched design at 2× would have had its two
hypotheses only **0.39 σ** apart — underpowered at n = 1. That is why the ladder is geometric (1× / 3× /
10×) rather than budget-matched: the rungs are chosen so the branches the falsifier decides are
separated by more than the noise, and the branches it cannot decide are labelled as such.

---

## §7 PRE-REGISTERED FALSIFIER — written before the fire

**Metric: the ENDPOINT flips vs the shared 33,757-flip init** (cadence-invariant per §4).
**Control: `EF3000` = −2,286**, identical steps / seed / curriculum / init / instrument.
**Band: σ = 856 flips.**

| `CW1-LR6E5` endpoint | reading |
|---|---|
| **≤ −4,000** (≥2.0 σ deeper) | **CONFIRMED — SCHEDULE-LIMITED.** Depth scales with LR budget; §5.2 is right and "sublinear return / interior best" was the anneal. **Fire rung 3.** |
| −3,142 … −4,000 (1.0–2.0 σ) | **PARTIAL.** Directionally LR-budget-limited, underpowered at n=1. **Fire rung 3** to extend the lever arm before concluding. |
| −1,430 … −3,142 (within ±1 σ) | **NULL.** Depth is not bought by 3× the LR budget — consistent with a real objective floor near −2,300. **Do NOT fire rung 3** on the depth question; route to the objective axis (`cos(sign g)` for unmeasured objectives) and to the transfer measurement. |
| **≥ −1,430** (≥1 σ worse, or positive) | **REFUTED at this formulation.** Higher lr damages under alignment as it did under misalignment; 2e-5 is at or above the aligned optimum and the sweep direction is **down**, not up. |

**Non-vacuity, checked before sealing:** the control (−2,286) sits inside the **NULL** band, so a null is a
live outcome and would genuinely close the depth line. The REFUTED band is reachable — the same trainer
at 2e-4 under the stock curriculum produced **+59,357** flips, so damage at high lr is demonstrated
behaviour, not a hypothetical. Both directions are real.

**INSTRUMENT-INVALIDATION conditions (result is void, not a verdict) — from the `frd077` lesson:**
* any endpoint of exactly **59,551,382 flips (50.48%)** — that is "predict Undrivable everywhere", the
  signature of a NaN frame absorbed by SegNet on MPS, and it is a constant wearing a measurement's clothes;
* any non-finite `quantized_exact_seg`, or `packed_parameter_bytes ≠ 40,252`;
* `rc ≠ 0`, or `resource_safe_run_status.json` `status ≠ "ok"`;
* `history` rows whose `phase` is not `expected_flip` at every logged step (proves the fractions took effect);
* `init_quantized_exact_seg ≠ 2.861616346571e-04` (proves the init is the shared one).

**What a CONFIRMED result would and would not buy.** It would say the aligned regime's depth is bought
with learning rate, on the semantic renderer, at n=1, on the trainer's advisory instrument. It would
**not** be a score, **not** byte-closed, **not** on the frontier vehicle, and **not** evidence that any of
it transfers. Say that when the row lands, rather than letting "the window is corrected" drift into "the
seg axis is solved."

---

## §8 The config-orphan — 5 flags of 38 now held, 33 still open

`lever_registry.completeness()` on `src/tac/pr130_lift/train_semantic_quantized_resumable.py` reported
**34 of 38 flags UNMAPPED**, including `--ce-fraction` and `--softplus-fraction` — the two flags that
produced the entire `ce1` result. Nine retained runs swept three decades of learning rate while holding
a curriculum shape that lived only in an argparse default, and that is exactly the config-orphan the
triality law names.

Landed here: `src/tac/witness_dsl/cw1_semantic_curriculum_levers_20260817.py`, declaring
`TRAINER_RELPATH = "src/tac/pr130_lift/train_semantic_quantized_resumable.py"` and three factories —

* `lever_cw1_aligned_objective()` → `--ce-fraction 0.0 --softplus-fraction 0.0`, custodied as a
  **MEASURED_ANCHOR** against the three-arm ladder (`semantic_curriculum_alignment_ladder_v1`);
* `lever_cw1_lr_budget(base_lr)` → `--lr`, custodied as a typed **class-4 HARDCODED_WAIVER** with
  `owner = ddm_cw1` and `rederivation_trigger = "two or more measured cells in the (lr × aligned-curriculum)
  column"` — because the aligned optimum is unmeasured and pretending otherwise is the constants-are-poison
  failure;
* `lever_cw1_observation_budget(eval_every, checkpoint_every)` → the cadence, **DERIVED_AT_CONFIG** from
  the measured 24.758 s/eval, and it **refuses** `eval_every == checkpoint_every` at construction, so the
  collinearity that produced a negative fixed cost cannot be re-introduced.

Verified: `package_lever_factories()` reports all three with `trainer_declared=True` and
`missing_flags=()` — every emitted flag exists in the real argparse (never-invent-flags passes). Tests
at `src/tac/witness_dsl/tests/test_cw1_semantic_curriculum_levers.py` cover the emitted flags, the
provenance rungs, the three refusals (`lr ≤ 0`, collinear cadence, cadence < 1), the package-registry
binding, and — as behaviour rather than constants — the two closed forms themselves, recomputed from
the source semantics inside the test.

**This closes 5 flags of 38. Thirty-three remain unmapped** (`--bits`, `--float-warmup-steps`,
`--distill-weight`, `--weight-perturb-*`, `--film-*`, `--carrier-*`, `--fixed-zero-mask`, the cache and
path flags, and the rest). I held the five the composed window is built from and the ones whose values
are load-bearing; I did not attempt the other thirty-three, and calling this orphan closed would be the
overclaim. The `#1092` drain levers in particular are still argparse-only.

**Honest limit:** `completeness()` reads only `curriculum_dsl.py`, so satellite lever modules — mine,
`hg1_ring0_margin_hinge_levers_20260816`, and `spec_tr1_renderer_20260728`'s factories alike — do not
register as `mapped` there. The package-wide `package_lever_factories()` sees them correctly. That
scoping gap is pre-existing and is not mine to fix in this unit; it is filed here so the next reader does
not mistake the unchanged `unmapped: 34` for a failed landing.

---

## §9 Retained payloads (bytes, not scalars)

| path | bytes | sha256 |
|---|---:|---|
| `/Volumes/APDataStore/pact/ddm_cw1/CW1_REDERIVATION.json` | 23,380 | `9dc49c75e8e176a0fc74e586173be8ffdc73af9ea0d0abc7a7b8fb9256862acf` |
| `/Volumes/APDataStore/pact/ddm_cw1/SEALED_TICKET_fire_lr6e5.sh` | 2,375 | `4c6e1f2a3ffdc5ef25e2a313e068737b082b2dd00f0079ffd73e208bea0af590` |
| `/Volumes/APDataStore/pact/ddm_cw1/SEALED_TICKET_fire_lr2e4.sh` | 2,376 | `eab9c617876555bcaf503bc798181afe0ac3e3bd8d8bbb645ea05a3f47826c91` |

`CW1_REDERIVATION.json` carries, per arm, the `result.json` path + sha256 + size, the full config, init /
endpoint / best in both seg and flips, eval and save counts, the `ema_policy`, and the launcher `elapsed_s`
— plus every claim's re-derivation and all six new findings with their `empirical_verification_status`.
Nothing was measured and discarded. `/Volumes/VertigoDataTier` was read-only throughout (it has 893 MiB
free); everything written went to `/Volumes/APDataStore/pact/ddm_cw1/`.

---

## §10 The sealed ticket — NOT FIRED

`bash /Volumes/APDataStore/pact/ddm_cw1/SEALED_TICKET_fire_lr6e5.sh` — then, only on a CONFIRMED or
PARTIAL branch, `SEALED_TICKET_fire_lr2e4.sh`.

**Dry-run validated** through `tools/launch_detached_process.py` (schema
`detached_local_process_launch.v2`; the launcher's system memory governor enforces the 116 GiB operator
ceiling through `safe_run`). Resource contract deliberately matches the prior arms exactly
(`mode: child_owned`, no thread caps) so the wall-clock instrument is unchanged — the one improvement is
`--projected-gib 3.1` from the **measured** peak RSS across five retained arms (2,831–3,087 MiB, max
3.01 GiB) in place of the guessed 4.0, and `--timeout 1740` against a 1,031 s prediction (1.69× headroom).

**A Metal fire was live at seal time** (`F1SIG1`, `--weight-perturb-robustness 1.0`, launcher counter 151,
pid 1746). One Metal fire at a time — MAIN must confirm it has finished before firing rung 2:

```bash
ps aux | grep -E 'train_semantic|train_ddm_cl1' | grep -v grep
```

---

## NEXT_IF_RESUMED

1. **Fire rung 2, gate on the falsifier, then rung 3 only on CONFIRMED/PARTIAL.** ~17 min each, $0.
2. **Aim `rg1b`'s `cos(sign g)` probe at the FRONTIER trainer.** §5.1 is the largest thing this unit
   found and it is one measurement from being actionable: the vehicle holding 0.029611 S of seg debt
   trains on token CE + rate and selects on bytes. Whether its gradient points where its score does has
   never been asked. This outranks rung 3.
3. **Second seed on `EF3000`.** Both the crossing and the interior best rest on n = 1; ~24 min.
4. **The transfer measurement**, which is where the S lives and which nothing here touches.
5. **Sweep `tau`** (§5.4) — the next inherited fraction-of-run constant, and the one the `ce1` line's own
   logic predicts is poisonous. It needs a flag first; that is a real code change on the lifted oracle,
   which carries pinned lift custody (commit `9049e1caa5`) and is not a drive-by edit.
6. **Do not quote `best` without its eval count**, anywhere, ever again (§5.6).

## §7.1 RUNG-2 RESULT — MAIN adjudication (2026-08-17, read AFTER §7 per the ticket)

`CW1-LR6E5` endpoint **0.000277684529622 = −1,000 flips vs init** (rc=0, 1,053 s, pid 36604,
payload `/Volumes/APDataStore/pact/ddm_cw1/LR6E5/`). Band: **≥ −1,430 → REFUTED at this
formulation.** 3× the integrated LR at identical steps/seed/curriculum/init lands 1,286 flips
(1.5σ — RE-GRADED to ~0.9σ vs the two-seed control mean, see §7.2) WORSE than the 1× control's −2,286. Higher lr damages under alignment as it did under
misalignment; **2e-5 is at or above the aligned optimum and any further sweep is DOWN, not up.**
Rung 3 (10×) does NOT fire, per the pre-registered gate. §8 item 2 (aim cos(sign g) at the
frontier trainer) was answered by `ddm_oa2` the same day: ∂d_seg/∂θ ≡ 0 there — no alignment
object exists; the frontier trainer is exactly rate-aligned by construction. Next from §8:
item 3 (second seed on EF3000 — fired at this adjudication) and item 4 (the transfer
measurement, where the S lives).

## §7.2 SEED-2 RESULT — the σ band is now MEASURED, and it re-grades §7.1 (2026-08-17)

`EF3000_SEED2` (seed 20260817, all other argv tokens identical, verified by set-diff at fire):
endpoint **0.0002745903862847222 = 32,392 flips = −1,365 vs init** (rc=0, 1,031 s, launch
counter 153, payload `/Volumes/APDataStore/pact/ddm_cw1/EF3000_SEED2/` — result.json +
per-200-step full-state checkpoints retained). `best_step` 3000 (at cap, same as seed-1),
`packed_parameter_bytes` 40,252 (byte-neutral ×3 now), EMA-deployed argmax parity 0/117,964,800
pixels, verdict PASS. Axis `[macOS-MPS training-signal]`, `score_claim: false`.

**The measured band.** Two real seeds of the IDENTICAL config: −2,286 (20260715) vs −1,365
(20260817) → |Δ| = 921 flips → σ_est = 921/√2 ≈ **651 flips** (n=2 — itself wide, ~76% rel.
error at this n). The previously QUOTED band (σ ≈ 605, 855 two-run) is **CONFIRMED in
magnitude** (ratio 1.08). No verdict flips on band size alone.

**The re-grade §7.1 owes.** The control's own seed spread (−2,286 … −1,365) **straddles the
pre-registered falsifier boundary (−1,430)**: seed-2's control endpoint lands INSIDE the
"REFUTED" band. The §7 bands were seed-fragile — calibrated against seed-1's control as if it
were the population mean. Re-graded rung-2 verdict (verdict_scope: formulation, unchanged):
**WEAKENED-DIRECTIONAL, not 1.5σ-REFUTED** — LR6E5 (−1,000) is worse than BOTH seed controls
(sign stable across seeds), but only **0.9σ vs the two-seed mean (−1,825.5)**. The ROUTING
conclusion survives intact: rung 3 stays cold, the un-swept lr direction stays DOWN — both
controls beat 3×, and nothing here licenses UP. What changes is the strength quote and the
lesson: **falsifier bands on stochastic endpoints must be calibrated against a seed ENSEMBLE,
never a single control run** (sister of the prefix-bias genus: one draw from a spread is not
the population).

**The other three verdicts re-checked against the measured band — all UNCHANGED:**
FRD077 seg-neutral (0.18σ → ~0.17σ at σ=651) · q3q4 lever −563 = 0.61 of the measured 921
two-run band, still within noise · F1 block (endpoint = init to 17 digits) is structural zero,
σ-independent.

## §7.3 — LR1E5D (lr-DOWN rung) adjudicated: NULL — the lr axis is CLOSED on this window

LR1E5D (lr 1.0e-5 = 0.5× baseline, seed 20260715, argv otherwise identical; rc=0, 1,219 s,
byte-neutral 40,252 B, best_step 3000 at cap, EMA parity clean) endpoint
quantized_exact_seg 0.00027326795789930556 → **−1,521 flips vs init** (init 33,757 → 32,236).
Against the SEALED ensemble-calibrated pre-registration (PREREGISTRATION.json, the first
consumer of `seed_ensemble_falsifier_band_v1`): inside the NULL band (−2,477 … −1,174),
**0.47σ from the two-seed ensemble mean −1,825.5** — indistinguishable from the controls.
**Verdict: NULL_2e-5_is_plateau** (verdict_scope: formulation — lr scaling of the aligned
expected-flip window on the semantic renderer, this init/schedule, single-seed rung read).
Both directions now measured: UP 3× = WEAKENED-DIRECTIONAL worse (§7.1–7.2), DOWN 2× = NULL
⇒ lr 2e-5 sits on a plateau; rung 3 (10×) permanently cold for this window; **the lr axis is
not the lever here**. Routing: §8 item 4 (the transfer measurement — where the advisory
training win lives in byte-closed S) is now the window family's head.
STORES CONSULTED: LR1E5D result.json + PREREGISTRATION.json (payloads retained per the law),
EF3000/EF3000_SEED2 controls, seed_ensemble_falsifier_band_v1.
