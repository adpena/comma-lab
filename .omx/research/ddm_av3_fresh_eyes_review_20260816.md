---
arm: ddm_av3
title: "fresh-eyes adversarial review of the 2026-08-16 arc: the seeded EMA-lag confound is REFUTED by direct checkpoint measurement, but the lr1 probe is confounded worse -- its own control C0 (lr 2e-7) reproduces 22% of A2's excursion and the trainer restarts Adam COLD at full lr with no warmup, so the ladder measures an optimizer transient, not a learning rate; A2 additionally computed its final payload and LOST it (no result.json, safe_run status=ok exit=1); and the day's convergence line drops rt1's own 'on the seg axis' qualifier over three live unraced rows"
utc: 2026-08-16
charter: ".omx/research/ddm_av3_fresh_eyes_review_against_all_charter_20260816.md"
axis: "[macOS-CPU advisory] + [macOS-MPS advisory] read-back of retained payloads -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-finding INSTANCE on the artifacts named in each row; family verdicts only where explicitly named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_av3 — fresh-eyes adversarial review against the whole 2026-08-16 arc

STORES CONSULTED (read at source, not from summaries): `ddm_rt1_seg_roundtrip_decomposition_20260816.md`
(§5, §6.1–§6.4, §7) · `ddm_td1_token_drop_schur_arithmetic_20260816.md` · `ddm_b2e_edit_replay_admission_verdict_20260816.md`
· `ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md` · `ddm_rc2_regime_charter_and_lr_probe_20260816.md`
· `src/tac/pr130_lift/train_semantic_quantized_resumable.py` · `src/tac/training.py::EMA` ·
`tools/safe_run.py` · `.omx/state/main_hot_state.md` · retained payloads
`/Volumes/APDataStore/pact/ddm_lr1/{A2,C0}/` (read-only) · memories [[m96]] [[m88]] [[m37]].

## ANSWER FIRST

**The seeded S1 hypothesis is REFUTED, and what replaced it is worse for the probe.** I loaded the
lr1 checkpoints and measured the shadow directly. The EMA is not lagging: `‖shadow − live‖ /
‖live − init‖` is **0.087 at step 100** and **0.008 at step 600**, and the warmup ramp has already
burned the init down to **2.2e-12** seed weight by step 100. A2's trajectory is real weight
movement.

**But the probe is confounded on a different axis, and its own control proves it.** C0 (lr 2e-7 —
100× lower) reproduces the same up-then-anneal shape: **+5,879 flipped pixels at step 100, +2,164
at step 600**, against A2's +27,170 / +8,049. A 100× learning-rate change bought only a **4.6×**
excursion. And **no arm ever beat step 0** — `best_quantized_exact_seg` stayed pinned at the init in
all four.

**With all four arms landed, the shape is a power law the control obeys too.** Across three decades
of lr and **250× of weight displacement**, `peak_flips ∝ ‖Δw‖^0.458` (**R² = 0.9969**). A smooth
local optimum would give exponent ≈ 2. Exponent ½ is what a **piecewise-constant argmax field
perturbed diffusively** gives: only pixels already within δ of a decision boundary can flip, so
flips scale as √δ. **The init sits on an argmax plateau, not in a descendable basin** — so the
adjudication's two branches ("genuine local optimum" vs "EMA artifact") are both wrong, and the
binding constraint is the objective's *direction*, not the learning rate. Full ladder in §F1b.

**The mechanism is an optimizer cold start, not a learning rate.** The trainer builds
`torch.optim.AdamW(model.parameters(), lr=args.lr, ...)` fresh (line 917) with **no moment restore
from `--init`**, and `CosineAnnealingLR` (line 918) with **no warmup**. Step 1 therefore runs at
full lr with zero Adam moments, where Adam's step is normalized to ≈`lr` per weight regardless of
gradient scale. The init is a checkpoint named `…tail6k_lr2e7` — a long lr-2e-7 tail. Every arm
kicks it off that point on step 1, and the excursion scales sub-linearly with lr exactly as a
normalized-step transient should.

**Three more things I could not verify as claimed:**

1. **A2 computed its final payload and lost it.** It crashed at `_atomic_torch_save(final_payload,
   args.save)` (line 1295) with `IsADirectoryError`, because the sealed ticket's `--save
   …/A2/checkpoints` collided with an existing empty directory. `_atomic_write_json(result,
   args.out)` is ordered **after** that save, so **A2 has no `result.json`** — the deployment-EMA
   `final_seg`, the argmax-parity receipt, `packed_parameter_bytes` and the verdict were all in
   memory and are gone. `tools/safe_run.py` reported **`status=ok` with `exit=1`**. A1 and A3 will
   crash identically if fired from the sealed ticket, which specifies that same `--save` path.
2. **`result.json` reports the INIT as the run's product, with `verdict: PASS`.** `best_state` is
   seeded from the step-0 EMA shadow (lines 969–972) and `best_key` includes step 0. C0 degraded at
   every single eval, fell back correctly — and its headline reads `quantized_exact_seg
   0.000286162`, `verdict "PASS"`. That is the input.
3. **rt1's η closure is a SOLVER-INSTANCE closure wearing FAMILY language, and rt1's own §6.1
   supplies the counter-evidence.** The shortfall to the bar is **0.1295 η**. The support-radius
   ladder in §6.1 swung η by **0.297** on one pair from a single untuned hyperparameter. §6.4's two
   named reopening conditions cover mechanism-collateral and byte-cost; neither covers solver
   quality — the one axis rt1 measured to move more than the shortfall.

**The convergence line is scope-widened in propagation.** rt1 §6.4 says it bounded every post-hoc
lever **"on the seg axis"**. `main_hot_state.md:114` reads **"All post-hoc levers bounded."** Three
rows are live and unraced underneath it: td1's 807 label-correction tokens (`verdict_scope:
INSTANCE (new, unraced)`), b2e's F3 `--film-row-dropout` (rc2's own "single never-fired lever",
covering 2 of the 3 refused edits), and the η closure above.

**What survives review unchanged:** rt1's decomposition arithmetic, its three self-caught defects,
its §7 negatives-accounting, the b2e collapse arithmetic (ratio-of-aggregates — correct under the
new pose-aggregation law), the rt1 η-gate pose fix (which landed in the **code and the n=9
artifact**, not just the memo prose — verified), and all three of rc2's premise corrections, which
I re-derived at source. The corpus sweep for other mean-of-ratios consumers found **two live
residuals**, neither on today's arc (F9).

---

## Ranked findings

| # | sev | finding | what it changes |
|---|---|---|---|
| **F1** | HIGH | EMA lag REFUTED (end lag 0.006–0.010 in all 4 arms); `peak_flips ∝ ‖Δw‖^0.458`, R²=0.9969 over 250× displacement — an argmax **plateau**, not a basin (a basin gives exponent ≈2) | answers MAIN's pending adjudication: neither branch is right; the constraint is the objective's direction, not lr |
| **F2** | HIGH | A2's final payload computed then discarded; no `result.json`; `safe_run status=ok exit=1`; sealed ticket reproduces the crash | A1/A3 must not fire as sealed; 2-landing fix owed (ALWAYS KEEP THE PAYLOAD) |
| **F3** | HIGH | `result.json` headline is the argmin **including step 0**, so a run that only degrades reports its INIT with `verdict: PASS` | any lr1 adjudication read from result.json top-level is unsafe; `history` is the only honest surface |
| **F4** | MED-HIGH | η closure is solver-instance, not family; shortfall 0.1295 < rt1's own measured 0.297 solver swing; at η=1 the channel supplies 76% of the gap | rt1 §6.4 needs a third reopening condition + headline aligned to §7 |
| **F5** | MED-HIGH | convergence line drops "on the seg axis"; 3 live unraced rows underneath | `main_hot_state.md:114` correction; the day is not "settled" |
| **F6** | MED | "lr 2e-7 barely trained" is the wrong diagnosis — measured, it trained **downhill** | b2e's caveat and rc2's binary framing both need a third branch |
| **F7** | MED | the LawRef-derived EMA decay is INERT for any run < ~1,167 steps; realized seed retention 3.3e-19 vs declared 0.01 | a `derived_at_config` constant is recorded as governing a run it does not govern |
| **F8** | LOW-MED | metric granularity is exactly 1/117,964,800; the probe's 1e-5 leg = 1,180 px, cleared by control-scale noise | the `>3×F(C0)` leg is the only load-bearing half of the bar |
| **F9** | LOW-MED | mean-of-ratios sweep: today's arc is CLEAN; two live residuals elsewhere (`ddm_pz1`, `ddm_et3`) + one reporting gap in rt1's per-run JSON | the new law needs one follow-up pass off-arc, not on it |

---

## F1 — S1: the EMA-lag confound, REFUTED at source, and what actually confounds the probe

### The measurement

`training_state.model_state_dict` (live) and `training_state.ema.shadow` (shadow) are both
retained in every `*.full_state.pt`. Init is the `--init` pin
`3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647`. L2 over all shared float
tensors, `‖init‖ = 74.4056`:

| arm | step | eff. decay | ‖live−init‖ | ‖shadow−init‖ | ‖shadow−live‖ | lag fraction |
|---|---:|---:|---:|---:|---:|---:|
| A2 | 100 | 0.91818 | 4.740e-02 | 4.632e-02 | 4.114e-03 | **0.087** |
| A2 | 200 | 0.95714 | 4.941e-02 | 4.952e-02 | 5.175e-03 | 0.105 |
| A2 | 300 | 0.97097 | 5.082e-02 | 4.898e-02 | 1.152e-02 | 0.227 |
| A2 | 400 | 0.97805 | 5.084e-02 | 5.067e-02 | 1.787e-03 | 0.035 |
| A2 | 600 | 0.98525 | 5.045e-02 | 5.051e-02 | 4.088e-04 | **0.008** |
| C0 | 100 | 0.91818 | 1.444e-03 | 1.380e-03 | 9.406e-05 | 0.065 |
| C0 | 600 | 0.98525 | 1.445e-03 | 1.445e-03 | 1.401e-05 | 0.010 |

`EMA.effective_decay()` returns `min(decay, (1+t)/(10+t))`. The warmup product telescopes to
`10!·(t+1)!/(t+10)!`, so the init's surviving weight in the shadow is **5.95e-05 at t=10**,
**2.15e-12 at t=100**, **3.29e-19 at t=600**. The shadow is not near the init at any eval point.

**Verdict: the #85 EMA-shadow-lag artifact is NOT present in the lr1 arms.** The warmup cure in
`tac.training.EMA` is doing exactly what its docstring claims. *verdict_scope: INSTANCE — the A2
and C0 arms, 600 steps, this trainer. It does not license a claim about other runs.*

### What replaced it

`quantized_exact_seg` is exactly `flipped_pixels / (600·512·384)`. Converted:

| step | A2 px | Δ vs step 0 | C0 px | Δ vs step 0 |
|---:|---:|---:|---:|---:|
| 0 | 33,757 | 0 | 33,757 | 0 |
| 100 | 60,927 | **+27,170** | 39,636 | **+5,879** |
| 200 | 47,994 | +14,237 | 37,667 | +3,910 |
| 300 | 45,766 | +12,009 | 36,380 | +2,623 |
| 600 | 41,806 | +8,049 | 35,921 | +2,164 |

Three things follow, none of which the adjudication assumed:

1. **The control is not null.** C0 at lr 2e-7 moves the judged metric by +17.4% transiently and
   +6.4% at the horizon. The "destruction then anneal-back" shape is a property of *starting
   training at all*, not of lr 2e-5.
2. **The excursion is sub-linear in lr.** 100× lr bought 32.9× weight displacement (clipping +
   cancellation) and only **4.6×** metric excursion.
3. **It is not annealing back.** A2's `‖live−init‖` goes 4.74e-2 → 5.08e-2 → 5.05e-2 — flat or
   growing — while the metric recovers 60,927 → 41,806. The optimizer is *rotating within a shell
   of roughly fixed radius*, finding a better point at the same distance from init. Reading the
   recovery as "annealing back toward the base" is wrong.

### The mechanism, verified at source

- `train_semantic_quantized_resumable.py:917` — `torch.optim.AdamW(model.parameters(), lr=args.lr,
  weight_decay=0.0)`. Fresh. `--init` restores weights only (rc2 confirms: "no optimizer state").
- `:918` — `CosineAnnealingLR`. Verified against the logged lr: at step 100, `2e-5·½(1+cos(π·100/600))
  = 1.866e-5` vs logged `1.8673651e-5`. **Pure cosine, no linear warmup.**
- Consequence: step 1 runs at full lr with zero first/second moments, where Adam's bias-corrected
  step is ≈`lr` per weight independent of gradient magnitude. Scale check: 80,504 params
  (`packed_parameter_bytes` 40,252 at 4 bits); `√80504 · 100·lr` = 0.57 (A2) / 5.7e-3 (C0);
  measured 4.74e-2 / 1.44e-3 — the same ~1/12 and ~1/4 fractions, consistent with clipping.

### F1b — the complete four-arm ladder (A1/A3 landed during this review)

All four arms measured the same way, from their retained `*.full_state.pt`:

| arm | lr | peak Δpx | end Δpx | recovery | ‖Δw‖@100 | ‖Δw‖@600 | max lag | **end lag** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2e-7 | 5,879 | 2,164 | 63.2% | 1.444e-03 | 1.445e-03 | 0.110 | **0.010** |
| A1 | 2e-6 | 14,960 | 3,828 | 74.4% | 9.521e-03 | 8.772e-03 | 0.159 | **0.006** |
| A2 | 2e-5 | 27,170 | 8,049 | 70.4% | 4.740e-02 | 5.045e-02 | 0.227 | **0.008** |
| A3 | 2e-4 | 76,594 | 59,357 | 22.5% | 3.614e-01 | 5.349e-01 | 0.271 | **0.009** |

**Answer to the pending adjudication: it is NOT an EMA artifact, and it is NOT a local optimum
either.** Two facts settle it:

1. **At the horizon, shadow ≡ live in every arm** — end lag 0.006–0.010. The four-arm ranking
   (C0 3.05e-4 < A1 3.19e-4 < A2 3.54e-4 < A3 7.89e-4) is a ranking of *live weights*. The lag
   peaks at step 300 in every arm — the ce→softplus_margin stage boundary, where the objective
   changes and the weights jump — and it collapses again by step 500. Real, explained, transient,
   and never at an eval that carries the verdict.
2. **The excursion is a clean SQUARE-ROOT law in weight displacement, across 250× of it:**

   > `peak_flips ∝ ‖Δw‖^0.458`  (R² = **0.9969**, four arms, three decades of lr)
   > `end_flips ∝ ‖Δw‖^0.557`   (R² = 0.9460)
   > `‖Δw‖@100 ∝ lr^0.789`      (R² = 0.9982)

**A local optimum in a smooth basin gives exponent ≈ 2** — the metric would rise quadratically and
the control would be nearly flat. Measured is **≈ 0.5**, and the control is *not* flat. Exponent ½
is the signature of a **piecewise-constant argmax field perturbed diffusively**: displace the
weights by δ and the number of pixels whose argmax crosses scales as √δ, because only pixels within
δ of a decision boundary can flip. That is a *plateau*, not a *basin*.

So the honest verdict is neither of the two branches the adjudication offered:

> **The init is not at a descendable optimum and it is not an EMA artifact. It sits on an argmax
> plateau where every displacement costs flips at rate √‖Δw‖, and the trainer's descent direction
> does not buy back more boundary pixels than the displacement costs — at any of the four learning
> rates.** A3 additionally breaks (22.5% recovery vs 63–74%) and its displacement is still *growing*
> at step 600 (1.48× its step-100 value), which is divergence, not a wider search.

This is a statement about the **objective**, not the learning rate — and it is the same conclusion
rt1 §6.4 reached from the other side. The metric counts one-pixel boundary crossings; the trainer
descends a scalar `curriculum_loss` that is already `.mean()`-reduced over the whole field
(`lifted/semantic_renderer_oracle.py:181`). No learning rate fixes a direction mismatch.

**Routing consequence for the lr1 adjudication.** "MOVED-UPWARD" is real but is not an lr finding —
every arm moves upward and the shape is a power law the control also obeys. The arms that would
discriminate the *remaining* question (is the plateau a property of the object, or of the cold
start?) are all one flag and cost ~380 s each:

- **W1 — lr warmup** (`--float-warmup-steps > 0`, already supported at `:1033`): if the step-100
  peak collapses while the horizon value holds, the peak was the cold start and only the horizon is
  the object.
- **R1 — resume optimizer state**: `--resume-from` restores AdamW moments (`:489`, `:1171`), so the
  run continues the tail instead of restarting it.
- **N0 — `--lr 0`**: the true null, isolating any non-optimizer drift. Cheapest of the three.

But note the ladder already bounds what they can find: the C0→A3 fit has R² 0.997, so a warmup arm
that lands off this curve is the interesting result, and one that lands on it closes the question.

*verdict_scope: INSTANCE (the four ddm_lr1 arms, burn-2 semantic base, 600 steps, MPS, seed
20260715, `--weight-qat-q3q4`). The √‖Δw‖ law is MEASURED on this ladder; the cold-start attribution
for the step-100 peak is DERIVED from code plus the scaling, not from an ablation — W1 is its
falsifier. Nothing here licenses a claim about longer horizons or other objectives.*

---

## F2 — A2 measure-and-discard, and a `status=ok` on a crashed run

Verified in `/Volumes/APDataStore/pact/ddm_lr1/A2/`: the directory contains the eight checkpoints,
`child.pid`, `launcher/`, `safe_run_status.json` — **and no `result.json`.**

The finalize block runs in this order (`:1206`–`:1296`): deployment-EMA re-eval → `final_seg` →
argmax parity → build `result` → `_atomic_torch_save(final_payload, args.save)` → *then*
`_atomic_write_json(result, args.out)`. The save raised:

```
IsADirectoryError: [Errno 21] Is a directory:
  '.../A2/.checkpoints.32551.tmp' -> '.../A2/checkpoints'
```

`_atomic_torch_save` (`:205`) ends in `os.replace(temporary, path)`; `path` was an existing empty
directory. So the JSON write never ran, and the deployment-EMA `final_seg`,
`ema_deployed_argmax_parity`, `packed_parameter_bytes`, `verdict` and the machine-readable
`history` were all held in memory and discarded. The run cost 379 s of Metal.

Three separate defects, each with a cheap permanent fix:

1. **No fail-fast validation.** `parse_args` validates `--steps`, `--float-warmup-steps`, batch
   sizes, cadences, `--ema-target-seed-fraction`, `--parity-pairs`, `--smoke-pairs` (`:819`–`:835`)
   — but never checks that `--save` is not an existing directory. A one-line `parser.error` there
   would have refused in milliseconds instead of after 379 s and a full final eval.
2. **Wrong write order.** The scalar result JSON is the cheap, irreplaceable artifact; the 1.7 MB
   checkpoint is the expensive, reproducible one. Writing the checkpoint first inverts the
   ALWAYS-KEEP-THE-PAYLOAD priority. Either write `args.out` first, or wrap the save so the JSON
   still lands.
3. **`safe_run` masks a crashed child.** `tools/safe_run.py:498` sets `status = "ok"` and only
   overrides it for `timeout` / `oom` / `killed` / `interrupted`. A non-zero child exit leaves
   `status` at `"ok"`. A2's receipt is literally `"status": "ok", "exit": 1`. Consumers that key on
   `status` alone will read a crashed run as successful.

   *Checked and CLEARED:* `tools/fit_ddm_cl1_hpac_capacity.py:308-310` requires
   `status == "ok" AND exit == expected_exit`, so that consumer is safe. I did not audit every
   watcher/closer; **could not verify absence of status-only consumers in the scope I read.**

**Blocking routing item.** The sealed rc2 ticket (§1, command shape) specifies `--save
/Volumes/APDataStore/pact/ddm_lr1/<arm>/checkpoints`. A2 fired that literally and crashed; C0
escaped only because it was fired with `--save …/C0/ckpt`. **A1 and A3 must not fire from the
sealed ticket until the `--save` path is changed**, or they will each burn ~380 s of Metal and
discard their results the same way.

---

## F3 — `result.json` reports the INIT as the run's product

`best_seg`/`best_state` are seeded from the step-0 EMA shadow *before the loop*:

```python
best_state = {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()}
history = [{"step": 0, "quantized_exact_seg": best_seg, ...}]     # :969-982
```

`best_key` is then compared against every eval (`:1126`). A run that only degrades never displaces
step 0. The final block re-evaluates `best_state` as the deployment EMA (`:1206`–`:1212`) and writes
`quantized_exact_seg = final_seg` into `result`.

C0's `result.json` therefore reads `quantized_exact_seg: 0.00028616163465711804`, `verdict: "PASS"`
— which is the init's number, unchanged, and `PASS` only because `0.000286 < 4e-4`. The run
degraded at all six evals. Nothing in the top-level result says so; only `history` carries it.

This is correct *selection* behaviour and misleading *reporting* behaviour. Cheap cure: add
`best_step` and `improved_over_init: bool` to the `result` dict. Until then, **no lr1 adjudication
may be read from `result.json` top-level fields** — `history` is the only honest surface, and for
A2 even that is only in `run.log`.

---

## F4 — S2: the η gate closes a SOLVER, and is written as closing a FAMILY

rt1 §6.4 opens: *"the post-hoc correction family is closed"*. §7 concedes: *"one solver budget, one
support radius."* Those are different claims, and the evidence supports the second.

**The shortfall is smaller than rt1's own measured solver sensitivity.**

- Bar 0.753, measured η **0.6235** (n=9) → shortfall **0.1295**.
- §6.1's support-radius ladder, on pair 34: r=0 → **0.5405**, r=1 → **0.6216**, r=2 → **0.3243**.
  A swing of **0.297** from one hyperparameter that was tuned on **n=2 pairs, unconstrained**.
- Before the support fix, η pinned at ~0. rt1's own sentence: *"A NO measured on sq1's support
  would have been an instrument artifact reported as physics."* The same sentence applies one level
  up: the current NO is measured on one untuned solver configuration.
- Untested solver axes: step budget (fixed 30), starts (fixed 2), `focus_weight`, solver lr,
  per-pair adaptive radius, non-integer support.

**The stakes make the scope matter.** Taking rt1's own numbers — seg gain at η=1 is
`0.0183/0.6235 = 0.029387 S`, rate cost `0.0221 S`:

| η | net S | fraction of the −0.0096 gap |
|---:|---:|---:|
| 0.6235 (measured) | +0.0025 (loss) | — |
| 0.753 (bar) | 0.000 | break-even |
| 1.00 | **−0.00726** | **76%** |

So the channel is not intrinsically small. Its entire verdict rests on η, and η is the quantity rt1
measured to be most solver-dependent. §6.4's two reopening conditions (a structurally
lower-collateral mechanism; a described set small enough that bytes fall faster than η) are both
correct and both *orthogonal to solver quality*.

**Checked and CLEARED — the bar arithmetic survives the coder-race move.** Break-even
`η = rate_cost / seg_gain(η=1)`:

| byte basis | rate S | break-even η |
|---|---:|---:|
| raced coder, 32,270 B (§5) | 0.021486 | **0.7321** |
| the memo's 0.0221 S ⇒ 33,190 B | 0.022100 | **0.7530** |
| §5 bar, 35,117 B | 0.023381 | **0.7966** |

Measured η 0.6235 is below **all three**, so the coder-race improvement does not flip the verdict —
the bar is ~0.02 conservative and the conclusion is robust to that. *One line of reconciliation is
still owed:* the 0.0221 S rate leg implies 33,190 B, which matches neither the raced 32,270 B nor
the §5 bar 35,117 B. I could not locate the derivation of 33,190 B in the memo.

**Correction owed (cheap, at source):** align §6.4's opening sentence with §7's scope, and add a
third named reopening condition — *a solver reaching η > 0.753 on the same support definition and
whole-frame accounting*. That keeps the closure honest without weakening it: the arithmetic still
says no fire-order today, and the 32,270 B coder is already built if it reopens.

---

## F5 — S5: the convergence claim is widened in propagation

`.omx/state/main_hot_state.md:113-114`:

> CONVERGENCE 08-16 (settled): rt1 #1075 CLOSED (eta 0.6235 n=9, non-supplier; pose-agg law
> banked) · b2e P1 refused-but-untested (lr 2e-7 barely trained). **All post-hoc levers bounded.**

rt1 §6.4's actual sentence: *"This unit has now bounded every post-hoc lever **on the seg axis**."*
The axis qualifier is dropped in propagation. Three rows are live underneath the widened claim:

1. **td1 H3 — the 807 label-correction tokens.** `td1:115` — *"verdict_scope: INSTANCE (new,
   unraced) for the 807-token label-correction set"*; `td1:132` — *"the one surviving candidate"*;
   `td1:168` — *"the qs family's cheapest remaining instance."* **Unraced is not bounded.**
2. **b2e F3 `--film-row-dropout`.** rc2's own ANSWER FIRST: *"The single never-fired lever is F3."*
   Two of b2e's three refused edits are FiLM row prunes and the lever that would train for them was
   off — confirmed in A2's `run.log` lever receipt: `"F3_film_row_dropout": {"active": false,
   "probability": 0.0}`. The editability thesis is **untested**, not bounded.
3. **The η closure** (F4) is solver-instance.

**A fourth instance of the same genus, inside rt1 itself.** Commit `bdc54e01d5` propagated n=9
through "three verdict_scope/limits lines still said n=4" — but **§6.4 still read "η = 0.6461
(n=6) … 0 of 6 pairs above it"** while the ANSWER FIRST carried 0.6235 / n=9. The routing section,
the one a reader acts on, was the stale one. Corrected at source in this unit (numbers unchanged,
n labelled; the derived 29,215 B threshold is left at its n=4 pooled η with the n stated, because
re-deriving it is a measurement I did not take). Three of the day's four stale-headline instances
were self-caught by the arm that made them, which is the reason the arc is trustworthy at all;
this one needed fresh eyes.

Also: rt1 bounds *seg* post-hoc levers. Pose post-hoc (the js8 gated-application line, declared
`DECLARED_UNBUILT_FOLLOW_ON` in the F5 lever receipt) and rate post-hoc (the coder race, which
**passed** at 32,270 B) are not in rt1's scope at all.

**Correction owed:** restore the axis qualifier and name the three open rows. The honest line is
*"post-hoc levers on the SEG axis are bounded; three rows remain unraced."* This is the same
stale-headline genus the day already caught twice (rt1's own n=4→n=9 propagation, and the
pose-aggregation correction) — worth noting that in both of those cases the arm caught itself.

---

## F6 — S4: "barely trained" is the wrong diagnosis

b2e reported ΔS_adv **+0.000336** and a **9-byte** weight-entropy change over 3,000 steps at lr
2e-7, and called the window untrained. C0 is the same trainer, same lr, same base, **1/5 the
steps** — and it moves the judged metric by **+5,879 px transient / +2,164 px (+6.4%) at horizon**,
with `‖Δw‖ = 1.44e-3`. b2e's own table agrees: the burn-2 base is *worse* than hv1 (S_adv 0.203144
vs 0.202808) and **330 B larger** (183,089 vs 182,759).

The two are consistent once the units are separated: 9 B is *packed* delta — very few quantization
codes flipped — while the float weights moved enough to flip thousands of boundary pixels, because
the flipped codes sit in FiLM/embedding tensors that touch the whole field. **The window trained
downhill.** That is a stronger statement than "did not train", and it changes rc2's framing.

rc2 §1 posed a binary: *"(a) the object is at a genuine optimum that no lr escapes, or (b) lr 2e-7
was ~2 orders too small."* The measured answer is a third branch neither arm enumerated: **every lr
tested leaves the init and none returns below it within 600 steps**, because the run restarts Adam
cold at full lr on a checkpoint produced by a long lr-2e-7 tail (F1).

**rc2's three premise corrections re-derived at source — all three hold.**

| rc2 claim | verified |
|---|---|
| `EditabilityLevers.applied()` is an in-forward straight-through operator | ✓ `editability_levers.py:331`; A2's run.log shows `F2_weight_qat_q3q4: {"active": true}` — the edit **was** in the loop |
| trainer does not refuse non-uniform grids; `--bits` is int4-packer-bound, F2 owns its own grid | ✓ `:819-820` (`--bits must be 4 because the deployed semantic packer is int4-only`) + separate `--weight-qat-q3q4` at `:777` |
| `--lr` default is 2e-5 at argparse line 738 | ✓ `:738` — `parser.add_argument("--lr", type=float, default=2e-5)`, line number exact |
| F3 `--film-row-dropout` is the one never-fired lever | ✓ present, default 0.0, receipt shows `active: false` |

rc2 checked code instead of memos and was right four times out of four. That discipline is the
reason this review found what it found rather than re-deriving it.

---

## F7 — the derived EMA decay is inert for every run shorter than ~1,167 steps

`resolve_ema_policy` (`:104`–`:155`) resolves `ema_decay_run_geometry_v1` at
`ladder_class=derived_at_config` from `updates_per_run=600`, `target_seed_fraction=0.01`, giving
**decay 0.9923540961321005** — the decay for which `0.99235^600 = 0.01`.

`EMA.effective_decay()` returns `min(decay, (1+t)/(10+t))`. The warmup branch reaches 0.99235 only
at **t = 1,167.1**. For a 600-step run the target decay is therefore **never applied on any step**,
and the realized seed retention is **3.29e-19**, not 0.01 — seventeen orders of magnitude from the
declared intent.

This is *not* the #85 bug: the warmup is the deliberate anti-freeze cure and F1 shows it working.
It is a provenance-honesty defect. `result.json.ema_policy` records the full LawRef manifest —
`equation_id`, `ladder_class: derived_at_config`, `value: 0.9923540961321005`, `fallback_used:
false` — as though it governed the run. It did not; the warmup ramp did.

**Cure:** have `resolve_ema_policy` also record the realized effective decay and seed retention
under warmup, and emit `warmup_dominates_target: true` when
`updates_per_run < (10·decay − 1)/(1 − decay)`. One derived field, and the manifest stops
over-claiming. *verdict_scope: INSTANCE for this trainer; the same `min(decay, warmup)` pattern is
in `tac.training.EMA` and may affect other short-run callers — **not swept, could not verify
scope**.*

---

## F8 — S6: instrument coherence

**Granularity.** Every logged `quantized_exact_seg` is an exact integer over `600·512·384 =
117,964,800`: 33,757 / 60,927 / 47,994 / 39,636 / 35,921 … Step size **8.477e-9**.

**The 1e-5 leg of the bar is not load-bearing.** `1e-5 × 117,964,800 = 1,180 px`. The *control's*
own step-100 excursion is 5,879 px — 5× the bar — and its horizon excursion 2,164 px is still 1.8×.
So the absolute leg is cleared by movement that the matched control also produces. The `Δ > 3×F(C0)`
leg is the only half doing work, and it is correctly relative. Recommend dropping or re-deriving the
absolute leg rather than reporting a "MOVED" that both legs did not jointly earn.

**The step-0 / advisory gap is NOT understood.** lr1 step 0 = 0.000286162; b2e's hv1 advisory base
d_seg = 0.00042714. Ratio 0.670. These are different instruments (lr1: PR135 semantic init vs
*transmitted tokens* at 384×512; b2e: hv1 archive vs *GT labels* at contest resolution), and rt1's
96.6% round-trip share does not obviously reconcile them. **I could not verify a receipt that
relates the two in the scope I read.** Treat the gap as open, not as understood.

**A near-coincidence to not cite.** lr1 step-0 is **33,757** flips; rt1's measured round trip is
**33,743** flips — 0.04% apart. They are different objects on different denominators. This should
not be cited as cross-instrument corroboration; it is a coincidence until someone measures the join.

---

## F9 — S3: the mean-of-ratios sweep

The new law (memory `pose_aggregation_is_mean_of_dpose_never_mean_of_ratios_20260816`) says the
scorer averages per-pair **d_pose values** then takes `sqrt(10·mean)`; averaging per-pair **ratios**
over-weights pairs with tiny denominators. Swept `tools/`, `experiments/`, `src/tac/`, and the
2026-08 `.omx/research` memos.

**Today's arc is clean — both named targets verified at source, not from prose.**

- **b2e collapse** (`experiments/ddm_b2e_edit_replay_admission.py:436,456,470`):
  `collapse = calibration_excess / burn_excess`, where each excess is built from **scalar evaluator
  aggregates** (`base_pose`, `edited_pose` read from `contest_auth_eval.json`). No per-pair array is
  in the path. **Ratio-of-means — CORRECT.** The 0.75–1.06× vs 50× verdict is not exposed.
- **rt1 η gate** (`experiments/ddm_rt1_eta_gate_pose_constrained.py:330-343`): η itself was never at
  risk (pooled at `:244`/`:317` as `(before.sum() − after.sum()) / desc.sum()`). The pose leg now
  computes `agg_ratio = da.mean() / db.mean()` and demotes the old statistic to
  `diagnostic_delta_S_pose_at_{median,mean}_of_ratios`. **Confirmed in the artifact**, not only the
  code: `eta_gate_null/ETA_GATE_VERDICT_AGGREGATE.json` has `n_pairs_landed: 9`,
  `d_pose_aggregate_ratio_scorer_convention = 0.7134220362199388` (the memo's ×0.713) and
  `delta_S_pose_scorer_convention = −0.00128862`. The n=9 numbers were produced by corrected code.
  The in-code comment even records that the two statistics **disagree in sign** here (1.809 vs
  0.431) — which is why the correction mattered.

**Two live residuals, both off today's arc:**

1. `experiments/ddm_pz1_dpose_window_solve_paired.py:127-131` — emits **both**
   `ds_pose_mean = CX1_POSE_CONTRIB·(√(sol.mean()/base.mean()) − 1)` (correct) **and**
   `ds_pose_median` built from `np.median(ratios)` (mean-of-ratios shaped), with a comment noting
   the two disagree under skew. Honest, but a consumer can quote the wrong ΔS. The pz1 memo
   headlines the correct one (1.4261) and flags the median (1.5453), so the memo is safe; the
   **artifact field is the exposure**. Highest-ranked follow-up.
2. `experiments/ddm_et3_solve_within_cvp_phase_field.py:486-487` — `pose_pass` gates partly on
   `abs(median(ratios) − 1) ≤ tol`. Conservatively bounded by the `AND ratios.max() ≤ thr` term, so
   it cannot pass on a skew artifact alone, but the median leg is the wrong statistic.

**One reporting gap on today's arc (minor):** rt1's **per-run** `ETA_GATE_VERDICT.json` (`:276-279`)
still emits a pose block of `d_pose_ratio_median` / `_max` / `pairs_pose_improved` with no
scorer-convention aggregate. Its `bar_arithmetic` is seg+rate only so nothing is miscomputed, but a
reader quoting that file's pose block gets only per-pair order statistics. The **aggregate** file is
correct. One line to add the aggregate field.

Also swept and clear: no geomean/gmean aggregation of pose or seg per-pair ratios anywhere in
`tools/`, `experiments/`, `src/tac/` (the `geomean` hits are `length_sigma` gauge normalization).
Two stale hits (`tools/analyze_cpu_cuda_eval_drift.py:304-307`, 2026-05-14;
`tools/build_frame1_joint_safe_cone.py:220-221`, 2026-06-09) and one live-but-non-ΔS
(`tools/cathedral_autopilot.py:1340-1370`, mean of per-candidate pose ratios per substrate class)
are recorded but not on any current decision path. **Frozen `experiments/results/**/source_bundle/**`
copies were excluded from live classification; could not verify their consumers.**

## Routing implications

### (a) the lr1 adjudication

1. **Adjudicate as PLATEAU, not as optimum-or-artifact** (F1b). Both offered branches are refuted:
   the shadow tracks live to <1% at every horizon eval, and the excursion follows √‖Δw‖ with
   R²=0.997 including the control. The receipt-supported line is *"no lr descends because the
   trainer's direction is not the metric's direction; the init sits on an argmax plateau where any
   displacement costs flips at rate √‖Δw‖."*
2. **A1/A3 have already fired** — with `--save …/ckpt`, so they survived. **A2 remains the only arm
   with no `result.json`** (F2); its ladder row above is reconstructed from `run.log` + checkpoints.
   Any future arm must use `ckpt`, and the argparse guard is the durable fix.
3. **The remaining discriminator is W1 (lr warmup), not another lr.** The ladder's R²=0.997 already
   bounds what a fifth lr can show. W1 / R1 / N0 each cost one flag and ~380 s and test whether the
   step-100 peak is the cold start; an arm landing *off* the curve is the informative outcome.
4. **Read `history`, never `result.json` top-level** (F3): C0's headline is its own input with
   `verdict: PASS`, and the same will be true of every arm that only degrades — which is all four.

### (b) the ddm_rg1 regime charter

rc2 deliberately did not build `band_objective.py` until the probe answered. **The probe has now
answered in the way that most strongly promotes it.** The measured plateau law (F1b) says the
failure is a *direction* mismatch — a scalar field-mean objective descending against a metric that
counts one-pixel boundary crossings — which is precisely what a band-weighted per-pixel objective
changes. The charter's own gating condition ("new objective terms must claim a DIFFERENT landscape
vs this measured no-descent baseline") is now answerable: the baseline landscape is `√‖Δw‖` with
R²=0.997, so a band objective must break that exponent, and that is a cheap pre-registered bar. The one code delta rc2 identified — a per-pixel variant of
`curriculum_loss` (`lifted/semantic_renderer_oracle.py:181`, currently scalar `.mean()`-reduced) —
is what rt1 §6.4 independently routes everything to (edge-weighted Road↔Lane). It is now the
prerequisite, not the downstream: an objective that does not weight the one-pixel Road↔Lane
boundary is being asked to improve a metric that is 99.22% that boundary, from an init already
tuned on it. **Also fire F3 (`--film-row-dropout`) in the same charter** — it is one flag, it covers
2 of 3 refused edits, and it is the largest never-fired surface in the editability thesis (F5).

### (c) the convergence claim

Amend `main_hot_state.md:114` to restore rt1's axis qualifier and name the three unraced rows
(td1's 807 tokens, b2e F3, the η solver axis). "Settled" is premature; "seg-axis post-hoc levers
bounded, three rows unraced" is what the receipts support.

---

## Corrections APPLIED at source by this unit

Per the stale-headline law, each correction touches the headline, the body, and the ledger row
together. No measured number was mutated; every correction is additive and labelled.

| file | what changed |
|---|---|
| `.omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md` §6.4 | heading "family is closed" → "closed AT THIS SOLVER"; stale n=6/0.6461 → n=9/0.6235 with the n=6 snapshot retained; "two named things" → "three"; the **third reopening condition** (a solver reaching η > 0.753) written out with §6.1's own 0.297 swing as its evidence; the 29,215 B threshold left at its n=4 η with the n stated |
| `.omx/research/ddm_rc2_regime_charter_and_lr_probe_20260816.md` §1 | sealed ticket `--save` changed `checkpoints` → `ckpt` (BLOCKING — prevents A1/A3 repeating A2's discard) + the incident recorded inline with the owed 2-landing |
| `.omx/state/main_hot_state.md` | CONVERGENCE row re-scoped: "settled / All post-hoc levers bounded" → "SEG-AXIS post-hoc levers bounded" with the three unraced rows named; two new rows for the A1/A3 block and the lr1 read correction |

**Deliberately NOT applied here** (they are `.py`, need the review-gate two clean passes, and an
audit arm should not land unreviewed code): the `--save`-is-a-directory argparse guard, the
result-JSON-before-checkpoint write order, the `best_step` / `improved_over_init` result fields,
the `warmup_dominates_target` EMA manifest field, and the `safe_run` non-zero-exit status. Each is
named with its file and line above; each is a two-landing item (fix + gate) for a build arm.

## What this unit did NOT establish

- **No ablation of the cold-start mechanism.** F1's mechanism is DERIVED from source + a scale
  check, plus the control's behaviour. N0/W1/R1 are the falsifiers and were not run (READ/AUDIT arm).
- **No re-run of A2.** Its `final_seg` is unrecoverable without re-firing; the step-600 stage
  checkpoint exists, so a resume-and-finalize is possible but was not done here.
- **No sweep of status-only `safe_run` consumers.** One consumer checked and cleared; **could not
  verify absence of others.**
- **No independent η measurement.** F4 is an argument from rt1's own published numbers about scope
  and stakes; it does not measure a better solver and does not claim one exists.
- **No claim the η channel would supply.** At η=1 it covers 76% of the gap; whether any solver
  reaches 0.753 is exactly the open question, and the arithmetic still says no fire-order today.
- **No pose-axis subset claim.** Per [[m96]] I drew no pose conclusion from any subset; the η pose
  leg's n=9 limits are rt1's own and I did not extend them.
- **No score.** Every number here is advisory read-back of retained payloads. The own-vehicle
  frontier is UNMOVED: hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`.
