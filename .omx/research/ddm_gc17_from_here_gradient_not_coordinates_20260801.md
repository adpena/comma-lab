# ddm_gc17_from_here — the coordinates are right; **two of the three gradients are wrong**

**Date:** 2026-08-01 · **Axis:** `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false`
**Cost:** $0, scorer-free — no n600 slot taken, no launch, no dispatch.
**Pointer:** UNMOVED. This document is MEANS, not progress.
**Convocation:** 18th. Schmidhuber LEAD. Full verb set.

> **HEADLINE, stated plainly per THE GOAL's means/ends firewall:** the exact pointer did not move
> today and this document does not move it. Its claim to value is that it **refutes one of my own
> seed's cures at source**, **refutes the operator's structural conjecture and replaces it with a
> better one**, and reduces "from here" to **two already-built, never-raced levers**.

---

## §0 PROVENANCE DISCIPLINE — the seed was working memory, and it was wrong in one place

Per the mid-flight binding amendment, every premise below was re-derived from a primary artifact.
Two of the seed's claims did not survive. Both corrections are in my favour of *less* work, not more.

| seed claim | source checked | verdict |
|---|---|---|
| v4d = 0.9639878 = 0.431179+0.292941+0.2398677 | recomputed | **CONFIRMED** 0.9639877; implies d_seg 0.00431179, d_pose 0.00858144, 360,238 B |
| bar 0.172141 | `.omx/state/canonical_frontier_pointer.json` | **PARTIAL.** SoT says `effective_frontier.score = **0.172**`, `score_precision: "official_display"`, PR130 `semantic-pose-HPAC_CPR1`, rank 1. **0.172141 is NOT in the SoT.** Treated as a replay figure below; conclusions use both and are insensitive. |
| GN capped at 2–3 relins by `__post_init__` validation | `terminal_pose_gn.py:497` | **CONFIRMED EXACTLY** — `_integer(..., minimum=2, maximum=3)` |
| GN has no convergence test | `terminal_pose_gn.py:1032`–1106 | **CONFIRMED** — bare `for iteration in range(...)`, no `break`, no tolerance |
| `pair_index: int` ⇒ 600 independent solves | `:854`, `:871`, `:942` | **CONFIRMED** within this module |
| "still descending 13–23%/iter at stop" | `ddm_eg1_pose_gn_rehearsal_20260728.json` | **CONFIRMED EXACTLY** — −13.2% and −23.2%, both final steps `admitted=True` |
| **"central-difference Jacobian on a differentiable net" ⇒ use an analytic Jacobian** | `terminal_pose_gn.py:1038-1049` | **❌ REFUTED.** The step is **±1 in int16 coefficient codes** — the *minimum representable step of a genuinely discrete variable*. It is an exact lattice secant, not a noisy derivative. An analytic Jacobian would be the Jacobian of a **relaxation**, i.e. of a different problem. **The seed's cure was wrong.** |
| **"raise the relinearization cap"** (my own developing thesis, ~40 min in) | `ddm_p3v2_optimal_form_pose_resolve_20260729.md` §2 | **❌ REFUTED BY AN EXISTING MEASUREMENT.** Run to ~11 relins the rank-6 cosine basis **plateaus** at d_pose mean 15.29 (trajectory 89.5→22.1→16.4→15.8→15.3→15.15→15.07→15.07). The cap is **not** the binding constraint. **The basis is.** |

**Before/after, declared as instructed.** I was ~40 minutes from recommending "remove the cap at
`terminal_pose_gn.py:497`" as action #1, with an extrapolated ~0.21 S. A measurement that already
existed on disk (`p3v2`, 2026-07-29) says the cap is second-order behind rank-deficiency. **The
recall was the lever, not the build** — exactly `[[corpus_first_and_the_recall_instrument_was_down]]`.

**Denominator honesty.** The charter asks for ΔS as a fraction of gap *and* of known inventory. The
gap denominator is computable. **The inventory denominator is not**: `current_focus` records that
only **5 of 179** levers carry a ΔS estimate. I report the gap fraction and state that the inventory
fraction is unavailable rather than invent one.

---

## §1 THE BACKCAST — the format is an excellent FALSIFIER-GENERATOR and a **terrible RANKER**

Audited gc5–gc10, **N = 39 recommendations**, each traced to build/measurement evidence or an
exhaustive negative search. The result refutes *both* of the charter's two candidate diagnoses.

| bucket | N | share |
|---|---:|---:|
| **BUILT + MEASURED** | **27** | **69%** |
| — measurement **REFUTED the convocation's own prediction** | **13** | 33% of all |
| — prediction survived | 10 | 26% |
| — measured but confounded / uninterpretable | 1 | |
| — apparatus, no ΔS applicable | 3 | |
| BUILT-NOT-MEASURED | 1 | 3% |
| NOT-BUILT | 8 | 21% |
| SUPERSEDED / RESCOPED after landing | 3 | 8% |

**Execution is not the bottleneck: 72% got built.** The falsifiers were real and fired fast, usually
within 24–48 h. And yet:

> **Six convocations, ZERO pointer delta.** `0.1910828242` in gc5's memo (07-28); `Pointer UNMOVED`
> in `ddm_cr1_...20260801.md` (08-01).

Two facts explain it, and neither is "bad ideas" or "poor execution":

**(1) Every single #1-ranked lever was refuted by the measurement it ordered.** gc6's P3 (−27.8 S
predicted → falsifier fired); gc7r/gc8's wr1 waterfill (−0.24…−0.28 predicted → **net +0.153**
measured); gc8's QA39, stamped "THE WINNER" (dead in <24 h); gc8's pfs1 D2 (falsifier fired); gc9's
Gate-B (S **3.3122** vs ref 2.2566); gc10's ν-bank ("nothing measured comes within 5× of that rate"
→ bank **empty**). Six for six, at the head of the ranking.

**(2) The bookings used to rank are ~100× optimistic, and the survivors were all downward
corrections.** QA03 booked −0.046…−0.138, realized **−0.001582** — 1.15% of the ceiling, 1.45 B/flip
≈ water against `W = 1.2731`. Of the 10 predictions that *survived* contact, **none produced a score
move.**

**Diagnosis.** The convocation ranks candidates **by predicted ΔS**, and predicted ΔS has a measured
track record of being ~100× overstated and **refuted-at-the-head six times out of six**. The format
is not generating bad candidates — it is generating *good falsifiable ones* and then **sorting them
with a broken estimator**. Ranking is the defective organ, not ideation and not execution.

**A second, distinct defect — retrieval — is visible in today's four refutations**, all resolved by
reading an artifact or source file that *already existed*, none needing a new measurement:

| claim | closed by |
|---|---|
| pose "photometric wall" (N1=NO), load-bearing in CLAUDE.md doctrine | `p3v2` (3 days old) |
| `#853` "−0.0499 S pose-independent rate half" | re-read of the byte fields |
| `#855` "MLX routing flips 76 px" | re-read against the torch authority |
| **my own "raise the GN cap"** | `p3v2` §2 (3 days old) |

The corpus corroborates that this is chronic, not a bad day: **QA22 was recommended 3× (gc6, gc7r,
gc8) and never run; QA17 Modal 2× and never flown; gc9 row 4 re-ordered a ν-measurement that had
landed the previous day.** gc5 is never accounted for by any successor.

### §1b The second half (gc11–gc17, N=47) — the format DEGRADED, and the trend is the finding

| | gc5–gc10 | **gc11–gc17** |
|---|---:|---:|
| recommendations tabled | 39 | **47** |
| BUILT + MEASURED | 27 (**69%**) | **18 (38%)** |
| NOT-BUILT | 8 (21%) | **23 (49%)** |
| own prediction measured FALSE | 13 | 12 |

**Execution nearly halved while volume grew.** The collapse localises: **gc14 onward tabled 28 rows
and produced 5 built+measured — of which 2 were already done *before* the convocation recommended
them.** `gc17` diagnosed *"retrieval is the binding constraint"* and then **exhibited it**: its
rank-1 rested on "`#826` has never been exact-eval'd," refuted by a receipt sitting in custody
**5 h 27 m earlier**.

**Combined, gc5–gc17: N = 86 recommendations, 45 built+measured (52%), 25 refuted their own
prediction, and ZERO pointer delta across thirteen convocations.**

**One correlation is actionable.** The *only* convocation whose recommendations produced measured
score-lowering rows on our own vehicle is **gc12** (rung-1 ep641 Δ **−0.0651** S_add; burn-4 w02
**−0.018303**; endpoint ep854 byte-closed **0.634232**, −0.013558 vs control) — and **gc12 is the one
convocation that mechanically audited its predecessor's table** (BR-A FALSE / BR-B DEAD / BR-D
FIRED). Auditing the predecessor is the only observed correlate of producing a real row.

> **VERDICT. Do not convene to RANK. Convene to FALSIFY — and rank by cost-to-falsify, never by
> predicted ΔS.** In the first half the estimator that sorted the table failed six times out of six
> at the head; in the second half the table stopped being executed at all. A convocation whose first
> act is generating candidates is means-as-ends. Its first act must be **(i) a mechanical audit of
> the predecessor's table and (ii) a recall pass naming, per candidate, the artifact that last
> touched it and its verdict-scope.** Today that pass alone refuted four claims at $0, two of them
> mine — and the audit found a finished measurement nobody had read (A0 below).

**This is why §5 ranks by cost-to-falsify, carries no ΔS at the head, and has three entries rather
than fifteen** — one of which is not a proposal at all but a harvest.

*Scope caveat inherited from both arms:* `/Volumes/APDataStore/pact`, `.omx/tmp/codex_worktrees/`,
and unmerged worktree branches were **not** searched. Every "NOT-BUILT" above carries that blind spot.

---

## §2 THE STATE, RE-DERIVED — and a harsher framing than "5.6×"

```
v4d   = 0.431179 (seg) + 0.292941 (pose) + 0.2398677 (rate) = 0.9639877
        d_seg 0.00431179 · d_pose 0.00858144 · 360,238 B
PR130 = 0.172141 (replay figure; SoT display = 0.172) @ 190,952 B ⇒ rate 0.127147
```

Split it the way that actually constrains us — **distortion vs rate**, not three axes:

| | ours | PR130 | ratio |
|---|---:|---:|---:|
| **distortion (seg+pose)** | **0.724120** | **0.044994** | **16.09×** |
| **rate** | 0.239868 | 0.127147 | 1.887× |

**85.8% of the 0.7918 gap is distortion.** We spend **89% more bytes** than PR130 and get **16×
worse distortion**. We are not at a worse point on the same R–D curve; **we are on a different,
much worse curve.** Rate-shaving cannot reach the bar: zeroing the *entire* rate gap leaves 0.679.

**The exchange rates, re-derived from first principles (not recalled):**

```
scored px = 600 pairs × 512 × 384                    = 117,964,800
S per pixel flip = 100/117,964,800                   = 8.477105034722222e-07
S per byte       = 25/37,545,489                     = 6.658589531221714e-07
W = BYTES PER FLIP                                   = 1.273108215332031
```

This reproduces the corpus constant to all printed digits. **W is the Chou–Lookabaugh–Gray
entropy-constrained-quantization Lagrange multiplier for this score**, exact and known: *spending
more than 1.2731 bytes to fix one pixel flip is a net loss.* It is an admission rule, not a
bookkeeping constant.

---

## §3 THE OPERATOR'S STRUCTURAL CONJECTURE — **REFUTED**, and replaced with something better

**Conjectured:** rate and distortion are one object + pose is pure realization ⇒ the `(seg, pose,
rate)` decomposition may be the wrong coordinate system, and waterfilling *between* non-independent
axes has been mis-pricing every decision.

**Refuted in three steps, each on primary source.**

**(a) There is no coordinate ambiguity.** `S` is *already* a scalarization of a single field θ with
fixed, exactly-known coefficients. The exchange rates are not estimated — they are closed-form (§2).
"Wrong coordinates" would require the objective to be ambiguous; it is not. What waterfilling
substitutes for is a *computable gradient*, and that is a different disease.

**(b) The vehicle DOES separate — along an axis the `(seg,pose,rate)` split hides.** Verified at the
frozen scorer, `upstream/modules.py:108`:

```python
x = x[:, -1, ...] # Use only last frame
```

**SegNet reads frame_1 only. frame_0 therefore carries exactly zero d_seg obligation.** PoseNet
reads both. So the true factorization is by **frame**, not by axis:

> **frame_0 = a pure pose actuator at zero seg risk, where only BYTES bind.
> frame_1 = the coupled seg+pose surface.**

This is granted by the frozen scorer's own architecture — a measured structural fact, not a modelling
choice. `(seg, pose, rate)` coordinates *conceal* it; `(frame_0, frame_1) × rate` *exposes* it. The
conjecture is right that our coordinates mislead — but **not because everything is coupled; because
the one place that genuinely decouples is invisible in them.**

**(c) The actual defect is the GRADIENT, and two of its three components are wrong.** Verified at
`experiments/train_tr1_partition_renderer_mlx.py:1936-1948`:

```python
acc = acc + cfg.w_rate * token_rate_term(mdl, ids)     # w_rate = 0.05
# pair_loss(...) passes w_pose=0.0, compute_pose=False   (:1851-1853)
```

| term | coefficient in the loss | score's true coefficient | status |
|---|---|---|---|
| seg | `w_seg = 100.0` | 100 | ✅ **S-exact** |
| pose | `0.0` | `√(10·d_pose)` | ⚠️ **ABSENT by design** (#383 terminal staging) |
| rate | `0.05 × bits/token` | `25/37,545,489 per byte` | ❌ **GENERIC constant on an ANTI-CORRELATED surrogate** |

The rate leg is the live defect and it is measured, not suspected. `ddm_rsf1` (2026-08-01):

- On the **live** burn lineage (r1c ep504→640), Spearman ρ between the in-loop `entropy` surrogate
  and real shipped SMEVR bytes = **−0.7235**, CI [−0.943, −0.227], **excluding zero on the negative
  side**. The surrogate fell −1.80% while shipped bytes **rose** +1.36%.
- On bc1 ep109→399: surrogate **−6.79%** while bytes **+28.9%** (194,236 → 250,358 B).
- Adjacent-step sign agreement with real bytes: **33–40%** — *worse than a coin flip*.
- `w_rate = 0.05` is classed **GENERIC-constant, no provenance rung** (`ddm_gd1` census row T19).

**The score's true coefficients have never once been inside a gradient.** They appear together in
exactly **two** places, and both are read-only: `ddm_composed_s_verdict.py:318` (a stage-exit verdict,
default-off, whose own docstring says *"VERDICT-LEVEL ONLY — NEVER differentiated through the burn's
training graph"*) and `src/tac/optimization/evaluator_action_waterfill.py:42-49` (a post-hoc
admission currency, `promotable: False`, not imported by the TR1 trainer). *Scope searched:*
`w_rate` / `rate_model` / `37545489` / `sqrt(10` / `w_pose *` across `experiments/`, `src/`, `tools/`,
`scripts/`, `configs/`, `.omx/`; **did not find** any differentiable objective carrying all three.
Seven sibling `*waterfill*.py` modules in `src/tac/optimization/` were **not** opened — a named
residual scope, not a cleared one.

Note also that `evaluator_action_waterfill.py:26-31` already records the non-additivity the
conjecture suspects — *"NONLINEAR + NONCOMMUTATIVE: SegNet/PoseNet are nonlocal + discontinuous, so
atom effects do not add… raw per-pixel waterfilling is wrong"* — so the corpus knew this and encoded
it **in the admission currency**, never in the gradient. That is the whole disease in one line.

**And the mechanism is a proof, not a tuning problem.** SMEVR's bytes live in its **value stream**
(69.4% of shipped bytes), priced by *mode-referenced residuals* — a purely **temporal** quantity. The
in-loop surrogate is a **marginal histogram entropy**, which is **invariant under temporal
permutation**. *A permutation-invariant functional cannot see a permutation-sensitive cost.* The
surrogate is structurally blind to exactly the structure the coder monetizes, and goes blind the
moment the field stops *filling* and starts *rearranging* — which is the regime the burn runs in.

> **REPLACEMENT THESIS.** The coordinates are right and the exchange rates are exact. What is wrong
> is ∇S: **one component is switched off, and one points backwards on the live lineage.** No change
> of coordinates repairs a wrong gradient. And the one genuine decoupling the vehicle offers —
> `frame_0` — is a *frame*, not an *axis*.

This is Schmidhuber's thesis stated as a defect: **bits are `−log P(next | context)`; we are
minimising `H(marginal)`, which discards the context.**

---

## §4 THE PANTHEON — derived from what the state needs, not re-summoned

Seats are justified by a named open question, per the charter's "derived, not habitual."

**N1 — a rate model that is a PREDICTOR with the coder's own context** (the §3c defect).
**Schmidhuber (LEAD)** — compression *is* prediction; the loss must reward predictability, not
marginal spread. He would invite **Solomonoff** (the ideal predictor), **Rissanen** and **Grünwald**
(MDL / predictive stochastic complexity), and above all **Willems–Shtarkov–Tjalkens (CTW)** — a
sequential context-tree mixture is *precisely* the object our surrogate should be and its mixture
weights are differentiable. **Ballé**, who would bring **Minnen & Toderici**: conditioning the
entropy model on causal context is the solved problem in learned compression, and we are on the
wrong side of it. **Chou–Lookabaugh–Gray (ECVQ)** for the classical statement of "one field, both
rate and distortion," whose multiplier is our exact `W = 1.2731 B/flip`. **Shannon** and **Berger**
hold the floor; **Blahut–Arimoto** for computing it.

**N2 — a pose actuation basis that is not generic** (the §5-A2 defect). The measured failure is
**rank-deficiency with respect to the pose Jacobian**, so the right object is the **Jacobian's own
row space**. **Eckart–Young / Golub–Van Loan** (optimal low-rank is the SVD of the operator you
actually care about, never a fixed dictionary). **Levenberg** and **Marquardt** — our solver has
Levenberg's damping frozen at `1e-3` and Marquardt's *adaptation deleted*. **Triggs** (bundle
adjustment: exploit Jacobian structure), **Barfoot / Chirikjian** (continuous-time SE(3) — the 600
solves are independent and the trajectory is smooth), **Dellaert** (factor graphs).

**N3 — the shared 1,565 px.** **Kahan**, **Higham**, **Demmel**.

**N4 — the format question (§1).** **Feynman** (cargo-cult science: the ritual is performed
faithfully and the planes do not land), **Meehl**, **Ioannidis**, and **Deming/Shewhart** — whose
*tampering* is the exact name for adjusting a process in response to noise you have not first
measured. Convening on an un-recalled corpus is tampering.

**What the pantheon would say, jointly, in one line:** *you are running an entropy-coded system whose
training signal cannot see entropy coding, and steering a 6-parameter solve through a dictionary
chosen for its name.*

---

## §5 WHAT TO DO FROM HERE — two levers, both already built, neither ever raced

Ranked by expected ΔS per unit wall-clock. **Neither requires new machinery** — this is the
"pay the debt on the EXISTING surface" discipline, and both are one-flag or one-array changes.

### A0 — **HARVEST `#824` (`ddm_bp1`). It is already measured. Nobody has read it.** Cost ≈ 0.

The backcast found a **completed n600 run whose result has never been landed**. `ddm_bp1` — gc14's R1
and gc15's rank-2, billed as *"the cheapest decisive arm in the campaign"* — ran to completion on
07-31 (receipts 16:23 / 16:59, custody `/Volumes/VertigoDataTier/pact/ddm_bp1_20260731/`), both arms
from the **same parent ep946**, argv identical except `--adam-bias-correction`:

| arm | n600 `full_confirm` d_seg | vs parent 0.004148441 |
|---|---:|---:|
| **A — incumbent (bias-correction OFF)** | **0.004134369** | **−0.0014072 S** |
| B′ — bias-corrected | 0.004259211 | +0.011077 S |

**Arm A beats its own parent, and the bias-corrected arm LOSES by 0.012484 S seg.** No memo, no
commit, no ledger row; `#824` is still `pending`. This settles gc15's H1 in the *opposite* direction
to the one feared (the descent is **not** a bias-correction artifact) — a real result, sitting unread.

- **Cost:** a memo + a ledger row. No compute, no slot.
- **Stakes:** −0.0014072 S = **0.18% of the gap**. Small — *and it is finished*, which is the point.
- **Why it is ranked first:** per §1's verdict, harvesting a completed measurement strictly dominates
  convening about an uncompleted one. This is grade-5 orphan debt
  (`[[designed_stub_is_orphan_signal_and_a_no_fake_violation]]`), found by audit, not by ideation.

### A1 — race `rate_model=smevr_surrogate` against the live `entropy`. **Highest yield, lowest cost.**

The trainer **already implements both modes** (`token_rate_term`, `train_tr1…:1895-1916`). rsf1
measured `smevr_surrogate` at ρ **+0.7412 / +1.0000** where `entropy` scores **−0.7235 / −0.5382**,
and its slope premise holds (`+120,239 B per bit/token, R² 0.84`) where entropy's has the **wrong
sign** (`−45,228`). Its DSL rung reads **`UNRACED (QA86a OWED)`**.

- **Falsifier:** if, at matched d_seg, shipped SMEVR bytes do not fall, the correlation finding does
  not transfer from *trajectory* to *descent direction* — scope INSTANCE→FORMULATION, not family.
- **Cost:** one burn A/B + a byte-close. No new code.
- **Stakes:** the rate gap is 0.112721 = **14.2% of the 0.7918 gap**. But this bounds it *only if the
  gradient is otherwise correct* — the term is inside the descent, so it also steers where seg goes.
  I will not put a ΔS on it; the honest statement is that a term measured to point backwards is
  currently multiplying every burn step.
- **⛔ ORDERING, from rsf1 and binding:** **do NOT raise `w_rate` 0.05 → 0.0768348 first.** The
  derived weight is derived against a premise the live surrogate violates; raising it would scale a
  backwards gradient by 1.54×. **Fix `rate_model` before re-deriving the weight.**

### A2 — build the pose actuation basis from the pose Jacobian's own SVD, on `frame_0`.

`p3v2` **MEASURED** that generic bases are the wall: rank-6 **cosine** converges (11 relins) to
d_pose **15.29**; `#715`'s **covariance-ordered** basis makes d_pose **RISE** with rank (19.89 @
rank-1 → 48 @ rank-6). Free unconstrained `frame_0` reaches **9.123e-5** (contribution 0.0302) — so
**the reach exists and the wall was an artifact**; the deficit is entirely *how we span it*.

The cure is already half-built: **`terminal_pose_gn.py:1035-1049` computes the `(6, rank)` Jacobian
every relinearization and then throws it away.** Its row space *is* the optimal rank-6 actuation
subspace (Eckart–Young), and it costs nothing extra to retain.

- **Falsifier:** if a Jacobian-aligned basis does not beat the cosine basis's converged floor at
  matched bytes, rank-deficiency is refuted and the residual is realization-limited elsewhere.
- **Scope guard (L18):** p3v2's numbers are on the **pb1-composed lineage (S≈20.27)**, *not* v4d.
  They do **not** transfer. v4d's pose contribution **0.292941 already beats p3v2's best cheap
  realization (≈1.98) by 6.8×**, so v4d's carrier is far better than that ladder. What transfers is
  the **mechanism** (generic bases are rank-deficient for the pose Jacobian), which is
  FORMULATION-scoped and vehicle-independent.
- **Stakes:** pose is **0.2776 S = 35.1% of the gap**, and reverse water-filling says the *bytes* are
  noise — the entire counted pose stream at PR130's d_pose is **1,325 B = 0.00088 S**, a **315×**
  ratio (arithmetic independently re-verified: θ=2.353e-5, bits/pair 17.67, 600 pairs → 1,325 B).
  **Pose is ~0% rate and ~100% realization**, and `frame_0` is a *seg-free* place to realize it.

### A3 — subordinate to A2, never before it: the three unladdered GN control knobs.

`relinearizations ≤ 3` (`:497`, validation-enforced, replicated in `ddm_su2_qa43_tail_solver.py:148,
:1819`, and the production driver `pb1_terminal_pose_gn_600.py:111` **defaults to 2**); `damping`
frozen at `1e-3` with **no Marquardt adaptation** — a rejected step loses the whole iteration
(`admitted=False`, 12 evaluations spent, no retry); `line_search = (1.0, 0.5, 0.25)` with validation
**forbidding any value > 1.0**, so the solver structurally cannot overshoot, while `np.rint` collapses
sub-half-unit steps to a silent no-op. Cost of one more relinearization = **12 evaluations** (rank 6).
eg1 shows both rehearsal solves terminated **by the cap while still admitting** (−13.2%, −23.2%).
**But the cap only binds if the basis has reach — which is precisely what A2 establishes.** Race
them together or not at all.

### What I would NOT do

- **Not** raise `w_rate` before `rate_model` (above).
- **Not** re-open reverse-waterfill `#766` as a parent — Knee-B is MEASURED-DEAD (S 3.3122 vs ref
  2.2566) — until A2 lands, because its rejection was attributed to pose damage that `ck1` then
  showed was **stale params** (recovery parity 0.98×). Its adjudication *depends on* the pose solve
  A2 changes. (Also: `#766` has **zero rows** in `canonical_task_status.jsonl` — unregistered.)
- **Not** chase the shared 1,565 px yet: `mi1` established every *reported* d_seg goes through
  torch-CPU, so it biases the training gradient only. Real, named, but behind A1/A2.
- **Not** build new machinery. Every action above is one flag or one retained array.
- **Not** convene `gc18` before a recall pass (§1).

---

## §6 VERDICT-SCOPE LEDGER

| claim | scope | authority |
|---|---|---|
| `rate_model=entropy` is anti-correlated with shipped bytes in the rearrangement regime | **FORMULATION**, regime-scoped | rsf1, exact byte columns |
| generic (cosine / covariance) bases are rank-deficient for the pose Jacobian | **FORMULATION** | p3v2 §2, n=6 / n=24 |
| the pose "photometric wall" | **REFUTED at INSTANCE** (naive solve artifact) | p3v2 §0 |
| GN terminates by cap while still admitting | **MEASURED**, n=2 solves | eg1 receipt |
| `frame_0` carries zero d_seg obligation | **STRUCTURAL** (frozen scorer) | `upstream/modules.py:108` |
| W = 1.2731082153320312 B/flip | **DERIVED**, closed form | recomputed §2 |
| 85.8% of the gap is distortion | **DERIVED** | §2 |
| the axis decomposition is the wrong coordinate system | **REFUTED** | §3 |
| geometric continuation of the GN descent | **CONJECTURE** — not claimed, not used | — |

**Pointer UNMOVED: v4d 0.9639877 `[macOS-CPU advisory]`; bar 0.172 (SoT display).** No score claim.
