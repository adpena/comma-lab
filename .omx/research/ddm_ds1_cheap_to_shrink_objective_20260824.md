# ddm_ds1 — the cheap-to-shrink objective: recall, derivation, mechanism race, build

**Date:** 2026-08-24 · **Arm:** ddm_ds1 · **Pointer:** UNMOVED · **R0 FIRED (§6b), zero new compute; R1 not fired.**
**Vehicle:** dx2 / hv1 / wd3 renderer · **Axis label:** design + build; no score claim.

---

## 0. Lead: my prediction was FALSIFIED, and the correction sharpens the arm

I predicted no `dD/dB`-shaped term had ever been **measured** on this lineage. **That is wrong, on
two independent grounds.** The first number I owe, per the charter:

> `experiments/ddm_wd3_scorer_aware_width_distillation.py:848-880`,
> `quantization_sensitivity_table` — *"First-order squared score effect for every parameter group
> and bit rung."* For every parameter group and every bit rung `b ∈ [2,8]` it computes
> `errors[b] = Σᵢ((wᵢ − q_b(wᵢ))·gᵢ)²` alongside `bytes[b] = 2 + ⌈numel·b/8⌉`.

That is a per-group `(ΔD, ΔB)` table, in the scorer's own gradient, **live in the production
trainer today.** Second ground: the corpus holds at least four *measured* shrink-damage ladders on
this lineage (§1.3). My prediction is FALSIFIED and I withdraw it.

**The precise claim that survives, and it is the whole arm:** every one of those objects **MEASURES**
`dD/dB` in order to **ALLOCATE**. None **MINIMIZES** it in order to **TRAIN**. The campaign has a
thermometer and no thermostat.

**The redirect, per the charter's instruction:** the unmeasured part is not the quantity — it is
whether the quantity is *movable*. That question is larger than one lever. tri1 closed 28 pairs and
56 triples, and fb1 closed the single-axis bound, **all priced at the exchange ratio of models never
trained to be shrunk.** If `dD/dB` is trainable at all, the entire closed composition surface was
priced on a constant that moves, and must be re-priced. If it is not trainable, the last route is
closed and the campaign should be told so plainly.

---

## 1. RECALL (first deliverable)

**STORES CONSULTED:** `.omx/research/*.md` (full grep) · `.omx/state/canonical_task_status.jsonl`
(+6 `.corrupt.*` snapshots) · `.omx/state/graph_memory/nodes.jsonl` ·
`.omx/state/canonical_equations_registry.jsonl` · `.omx/state/subagent_progress.jsonl` ·
`.omx/research/harness_tasklist_bridge_20260803.jsonl` · `docs/` · `tools/` · `experiments/` ·
`src/` · the PR130 intake code root
`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/` · `git log`.

### 1.1 The three named tasks: all OLD VEHICLE, all closed

**None of #242, #496, #111 exists in the repo ledger.** Not found in
`.omx/state/canonical_task_status.jsonl` (585 events, 228 distinct ids), its corrupt snapshots, the
harness bridge (which exports only window #800–#921), or `grep -rlE '"task_id":\s*(242|496|111)\b'
.omx/`. They are harness-only ids — the m89 split, exactly as the charter warned. What follows are
**repo mirrors**, cited as such.

| id | Mirror | Status | Lineage |
|---|---|---|---|
| **#242** MDL weight-compression regularizer | `codex_findings_ddm_ra1_..._20260724T230432Z_codex.md:82` | pending → **SUPERSEDED / CLOSED** (never "completed") | **OLD retired** — mirror's own words: *"old learned-weight vehicle"* |
| **#496** M+Adam low-precision rate | `operator_paper_intake_madam_..._20260713.md:11`; `ra1…:97` | **DOMINATED / INERT-CURIO** → CLOSED | **OLD retired** (witness_capstone_v3_n600) |
| **#111** Variable-grid QAT | `track_a_completeness_ledger_20260612.md:208` | **QUEUED, blocked-by #104** — never fired | **OLD retired** (witness track-A) |

**#242's flat-minima work was never measured.** `curriculum_candidate_pool_p0_20260710.md:107`:
*"#242 SAM/flat-minima MDL pre-quantization stage… design-memo-first… DERIVED (task #242 pending)"* —
no number, no receipt. My sub-prediction about #242 holds; the family was never tested.

### 1.2 The concept is already in the corpus — and it is two days old

`ddm_nt1_trained_at_target_rate_20260824.md:202-211` states it verbatim, including *"flatness in the
bit-depth direction"* and *"second-order where the current term is first-order"*, and self-grades at
`:231`: **"The 'cheap to shrink' objective (§3c) is unbuilt and unmeasured. It is a design
proposal."** I am not claiming the idea. I am claiming the derivation, the location, and the build.

### 1.3 Measured shrink-damage ladders on THIS lineage — all deeply underwater

| Object | Measured | Receipt |
|---|---|---|
| sa1 V-series bit ladder (sz1 = dx2 ancestor) | V7 q2: d_seg 0.05976591, d_pose 9.83209991, credit overwhelmed **675×**; V2 mildest rung **90× underwater** | `ddm_sa1_vseries_v7_dead_v2_routing_20260818.md:16-21,35-42,85-93` |
| WD4 **trained** w64 vs FX5 | `(−13,927 B, +0.03161884 d_seg, +13.43292362 d_pose)` ⇒ S 14.8829 ⇒ **~1,590× underwater** | `ddm_jx1_joint_exchange_envelope_20260822.md:100` |
| MP2 mixed-precision 4-rung vs HV1 | `(−823 B, +1.14e-6, +5.8376e-4)` ⇒ +0.365 S; three further rungs +0.336/+0.308/+0.256 | `ddm_jx1…:88-91` |
| keep-percent mass ladder | retention 10.47% → 35.19% | `ddm_keep01_ninth_pointer_move_verdict_20260818.md:30-36` |

**Law already established:** damage ∝ weight-mse^~0.4, which falls **slower** than rate credit at
every measured depth. There is no sweet spot on the existing curve. A cheap-to-shrink objective must
**change the curve**, not find a better point on it.

⚠ **One number needs a confound check before anyone treats it as the bar.** The WD4 1,590× row is
driven almost entirely by `d_pose +13.43` (√(10·13.43) ≈ 11.6 of the 14.88). That is *broken* pose,
not gracefully degraded pose, and wd4 carries `TRAIN_GATE_BLOCKED.json`
(`ddm_wd4_warm_lineage_width_20260821.md:73-90`). An unconverged run is an implementation-level
result, not a capacity law. **I flag it; I did not re-run it.**

### 1.4 Prior rate-regularizer attempts — the cautionary result

On the **old** vehicle, penalizing byte COUNT was catastrophic: Ballé weight-entropy λ=5 bought
−13% bytes for **+0.029 d_seg = +2.89 S worse**
(`measured_lever_inventory_for_synergy_pass_20260701T001751Z.md:156`); a λ=50 sweep cost **+3.80 S**
(`sweep_arm_A_scorerelevant_drain_20260714.md:77`). This is direct evidence *for* the reframe:
penalize the **sensitivity**, never the **count**.

---

## 2. DERIVATION — what `D` and `B` actually are

### 2.1 The archive splits in two, and only one half can host this objective

**Verified at source: the token codec is LOSSLESS.** `codec_hpac_integer.py:97-130` decodes with
`constriction.stream.queue.RangeDecoder`; `:165` offers *"fail unless decoded tokens exactly match
--cache"* and `:194` raises *"decoded tokens differ from the requested target cache."*

The consequence is structural and it decides the arm:

| Half | Bytes | `dD/dB` | Consequence |
|---|---:|---|---|
| **HPAC token model** | 70,453 | **≡ 0** — lossless; its weights cannot move distortion at all | Cannot host the objective. Its two loss terms (`F.cross_entropy` + `rate_lambda·log2·variable_weight_bits/pixels`, `train_ddm_cl1_hpac_capacity.py:1319-1322`) are **both bytes**. nt1's "rate is already in the objective" is not just true there — it is *complete*. |
| **Token-field edits** | payload | > 0 | Real, and measured (fs2/fs3/tba1/ec1). Data, not weights — no trainer owns it. |
| **Renderer (wd3)** | **30,856** | > 0 | **The only D-coupled *weights* in the archive.** |

Arithmetic check on the renderer figure: `0.14821987563243377 − 30,856 × 6.658590e-07 = 0.127674`,
matching fb1's independently-stated 0.12767 (`ddm_fb1_…:47`). The split is consistent.

**Note the irony, because it is load-bearing in §3:** the object whose removal operator has *perfect*
nesting (the PR130 `bit_depth` clip intervals `[−2^(b−1), 2^(b−1)−1]`, which genuinely nest) is the
object that carries **no distortion**. The object that carries the distortion has a **non-nested**
operator. We cannot borrow Matryoshka's nesting property; only its form.

### 2.2 `B` — exact, not modelled

Per parameter group, per rung: `bytes[b] = 2 + ⌈numel·b/8⌉` (`wd3:871`). The real coder's cost.
(On the HPAC side `B` is exactly linear in bit-depth: `variable_weight_bits = Σ_c relu(b_c)·n_c`, so
`dB/db_c = n_c` in closed form — `hpac_self_compress.py:84-98`.)

### 2.3 `D` — the contest distortion itself, never a proxy

`wd3:782-789`: `total = calibrated_seg + pose_score + duals·violations`, where `pose_score =
sqrt(clamp(10·pose_mse))` (`:776`) is **literally** the contest pose term and `calibrated_seg`
carries `seg_score_coefficient == 100.0` (asserted mandatory at `:457`). Both run through the real
frozen `posenet`/`segnet` graphs (`scorer_forward`, `:833-844`).

**This satisfies `#1127` by construction.** Render amplification ~38,700× means weight-MSE is the
wrong `D`; wd3's loss is the scorer's `D` in score units. Any mechanism that lives here is measuring
the right quantity.

### 2.4 The removal operator — exact, differentiable, and already in the loop

`quantize_tensor_groups` (`ddm_wd3_student_receiver.py:187-208`): per group, `scale =
max|w| / (2^(b−1)−1)` stored **fp16**, then `round(w/scale).clamp(±max)·scale`. The scale is
re-derived at every `b`, so **the `b` and `b−1` grids are not nested.**

`fake_quantize_state` (`:212-228`): `result[name] = live + (quantized − live).detach()` — a
straight-through estimator. `packet_quantizer_in_loop` is **asserted mandatory** (`wd3:449,460`), so
the renderer is *already* QAT-trained at its allocation.

### 2.5 The conversion from objective to bytes is already wired

The allocator (`wd3:884-949`) is a **discrete waterfill**: start every group at 2 bits, then greedily
upgrade the group with the best `saving/extra_bytes` until `total_error() ≤ maximum_predicted_error`.

> **Lower sensitivity ⇒ the waterfill terminates sooner ⇒ a cheaper allocation ⇒ fewer renderer bytes.**

The objective does not have to guess at bytes. `choose_cheapest_passing_quantization` converts
sensitivity into bytes deterministically, today. This is the causal chain, and none of it is new code.

### 2.6 The STE consequence — why a single-rung penalty cannot work

Because `fake_quantize_state` passes gradients through as **identity**, training at one allocation
gives the model **no gradient signal whatsoever about its sensitivity to bit depth.** The model
learns "be good at *this* allocation" and learns nothing about the neighbourhood. Only **evaluating
`D` at a genuinely different allocation** creates that pressure.

This is the derivation's load-bearing step: it rules out every single-point penalty and forces a
multi-rung evaluation. It is also why the existing in-loop QAT, despite being real QAT, does not
already deliver the property.

---

## 3. MECHANISM RACE

Reference forms cited per MAIN's charter correction. Deltas declared SCOPE vs MECHANISM.

### 3.0 The candidate that looks cheapest, and is a trap

**Penalize the allocator's own `total_error()` directly.** It is one line, it is exactly the quantity
the waterfill thresholds against, and it is **Goodhart**. The proxy is `Σᵢ((wᵢ − q(wᵢ))·gᵢ)²` —
first-order in the training gradient `g`. As training converges `g → 0` elementwise, so the proxy
collapses toward zero **at a sharp minimum exactly as fast as at a flat one.** Optimizing it teaches
the model to report a low predicted error, not to survive quantization. Agreeing with the test.

**This also implies a live defect worth reporting independently of this arm.** The allocator
thresholds a *shrinking* quantity against a *fixed* ceiling, so as training converges it should get
progressively more permissive and select ever-cheaper allocations whose true damage is unbounded.
I did **not** measure this — it is a hypothesis with a cheap test, pre-registered as **R0** in §6.

### 3.1 The four charter candidates

| # | Family | Reference form | Verdict |
|---|---|---|---|
| **(a)** | Sharpness-aware / flat minima | Foret et al. 2021: `ε* = ρ·∇L/‖∇L‖`, gradient taken at `w+ε*`; m-sharpness variant | **DOMINATED.** `ε*` is a linearized, isotropic, gradient-oriented stand-in for a perturbation **we can apply exactly**. MAIN is right that a generic `ρ≈0.05` is a toy here and that `ρ` would have to come from the coder's geometry — but once you derive the perturbation from the coder you have *reconstructed the exact operator*, and the surrogate has no remaining job. Substituting it would be a **mechanism reduction**. Second strike: a sharpness measured in weight space is the wrong curvature under `#1127`. |
| **(b)** | Nested / ordered | Kusupati et al. 2022 (Matryoshka): explicit weighted sum over a declared set. Sisters: Yu & Huang 2019 slimmable (sandwich rule; **switchable BN**), Horváth et al. 2021 FjORD (ordered dropout / sampled configuration) | **WINNER.** See §3.2. |
| **(c)** | Range-QAT | Esser et al. 2020 LSQ/LSQ+ (learned step size) | **MERGES INTO (b).** Our QAT is already in-loop at a fixed budget (`wd3:449,460`); "range" *is* "train over an ordered set of budgets", which is (b). LSQ remains the right citation for making the allocation *learned* rather than *selected* — a genuine successor, out of scope here. |
| **(d)** | Curvature along removal directions | Dong et al. HAWQ / HAWQ-V2 | **CORRECT THEORY, WRONG IMPLEMENTATION.** HAWQ diagnoses precisely why §3.0's first-order proxy is blind: at convergence the true damage is `½ΔwᵀHΔw`. But a Hessian estimate approximates a quantity we can **evaluate exactly** by applying the real quantizer. Using Hutchinson where an exact evaluation exists is a mechanism reduction (MAIN's own note). Its value here is diagnostic, and I have used it as such. |

### 3.2 The winner, and why the exact form dominates all four

**Advance (b) at reference form: an explicit weighted sum of the real score-aware `D`, evaluated at
an ordered set of progressively cheaper allocations.**

```
L = w₀·D(a₀) + w₁·D(a₁) + … + w_k·D(a_k)
```

`a₀` is the shipped allocation. `a₁…a_k` are produced by re-running **the trainer's own waterfill**
at loosened `maximum_predicted_error` ceilings — no new allocation policy is invented.

The race resolves on one fact: **we own the exact removal operator, it is differentiable, and it is
already in the training loop.** (a) and (d) are both approximations of `D(shrunk)`; (b) evaluates it.
When the exact quantity is affordable, every surrogate is a mechanism reduction.

Two precision points MAIN specifically asked for:

- **We borrow Matryoshka's FORM, not its nesting.** §2.4 shows the per-group fp16 scale is re-derived
  at each `b`, so the grids do not nest. This is an *ordered set of configurations*, which is exactly
  what slimmable/FjORD train over. Calling it "nested" would be a false claim.
- **Switchable BN is provably not load-bearing here.** Slimmable networks need it because
  per-configuration activation statistics diverge. The WD3 student uses `nn.GroupNorm`
  (`ddm_wd2_student_receiver.py:89,108,126`), which holds **no running statistics** and normalizes
  per sample. There is nothing to switch. This is also a reason to prefer **bit-depth** over **width**
  as the shrink axis: `:63` refuses widths incompatible with *"the inherited GroupNorm law"*, so width
  perturbs the norm topology while bit-depth does not touch it.

### 3.3 Declared deltas

- **SCOPE (legal):** `mode="sampled"` evaluates one rung per step instead of all `k`. The objective is
  **identical in expectation** (FjORD's estimator) and every evaluation is exact-in-`D`; only gradient
  variance changes. Small `k` (1–2) in the smoke.
- **MECHANISM: none.** `D` is the real loss through the real frozen scorers; the perturbation is the
  real packet quantizer; the byte cost is the real coder's.

---

## 4. WHICH TRAINER, AND WHY

**`experiments/ddm_wd3_scorer_aware_width_distillation.py`.** It is the only one of the three that
can host the term:

| Trainer | Owns `D`? | Owns `B`? | Verdict |
|---|---|---|---|
| `tools/train_ddm_cl1_hpac_capacity.py` | **No** — no scorer, renders no frames, and its codec is lossless so `D` is *constant* w.r.t. everything it controls | Yes (both terms) | **Cannot host it.** Not an oversight — a structural impossibility. |
| `experiments/ddm_wd3_scorer_aware_width_distillation.py` | **Yes** — `calibrated_seg + pose_score`, score units, frozen scorers | Yes — exact per-group `byte_costs`, and the waterfill that spends them | **The trainer.** |
| `train_semantic_quantized_resumable.py` | — | — | SISTER, EMA helpers only, not this lineage (`#1243`). |

---

## 5. THE BUILD

**`experiments/ddm_ds1_cheap_to_shrink.py`** — 14,796 B, sha256 `f58915f5891d1c53…`
**`src/tac/tests/test_ddm_ds1_cheap_to_shrink.py`** — 14,435 B, sha256 `857232102460d57d…`
**39 tests pass; `ruff check` clean.**

Design decisions, each with its reason:

- **Dependency-injected.** The waterfill and the per-rung `D` are passed in, so the module imports
  nothing heavy, needs no governed admission, and is unit-testable without the scorer.
- **Default-OFF, and inert in the strongest sense.** When inert, `apply` returns the caller's **own
  loss object**, not `loss + 0.0`. The autograd graph is unchanged, so byte-identity is structural
  rather than argued.
- **Resumability P0 with zero new state.** The rung schedule is a pure function of `(seed, step)` via
  SHA-256. Nothing is registered, so an existing checkpoint schema **cannot silently restart it** —
  the exact failure nt1 warned about at `:213-217`.
- **Fail-closed ladder.** `derive_rung_ladder` refuses any declared rung that is not strictly cheaper
  than the shipped allocation. A rung that saves no bytes teaches the model nothing; training on it
  silently would be an inert-lever bug.

**Binding-vs-inert proof** (`test_inert_returns_the_same_object`,
`test_inert_leaves_the_gradient_bit_identical`, `test_enabled_changes_the_value_and_the_gradient`):
off ⇒ same object and bit-identical gradient; on ⇒ value `9.0 → 45.0` and gradient `6.0 → 30.0` on a
closed-form case. The lever provably cannot move a byte when off, and provably moves the descent when on.

**Three defects I introduced and caught in my own review passes.** All three are the same genus —
a config that looks active while some part of it is silently inert — and all three are fixed
structurally rather than by documentation:

1. `mode="sandwich"` accepted `k>1` rungs while the step-selector only ever returned the cheapest, so
   middle rungs' declared weights would have been **silently unused**. Now refused at construction;
   `mode="all"` carries the full deterministic sum (`test_sandwich_refuses_more_than_one_rung`).
2. `base_weight=0.0` was accepted, which abandons the **shipped** allocation and trains only the
   cheap rungs — not a cheap-to-shrink objective at all, but it would have run and produced numbers.
   Now refused (`test_zero_base_weight_refuses`).
3. The cheaper-rung byte check was skippable and its absence invisible, so a skipped check looked
   exactly like a passed one — the vacuity-equals-pass genus. `diagnostics["byte_cost_checked"]` now
   reports the denominator (`test_ladder_records_whether_the_byte_check_ran`).

**39 tests pass; `ruff check` clean.**

**Not yet wired into wd3's step.** The call site is a ~10-line diff in a governed trainer. I am
handing MAIN the module, the tests, and the exact wiring point rather than editing a live governed
trainer unreviewed.

---

## 6. PRE-REGISTERED EXPERIMENT

**Exchange ratio** `r ≡ ΔD / (λ_B·ΔB)`, `λ_B = 6.658590e-07 S/B`
(`ddm_tx1_toolbox_crosswalk_20260819.md` §0, cited not re-derived). Win iff `r < 1`.

### R0 — allocator calibration audit (FREE, no training, run this first)

**Question:** does the allocator's first-order `total_error()` proxy actually predict true `ΔD`?
**Method:** on existing `fire_rung*` / `n120_confirmation` checkpoints, for several allocations,
compare predicted `total_error()` against **true** `ΔD` measured through the real scorers.
**Why first:** it is nearly free, it uses only existing artifacts, and it is decisive either way —
if proxy and truth diverge, `choose_cheapest_passing_quantization` is mis-ranking allocations today
and that is a byte-relevant bug **independent of this arm**. It also calibrates the rung ladder.
**Report both:** Spearman/Kendall on the **ordering** (a predictor can be badly scaled and still rank
perfectly, and **ranking is what the waterfill consumes** when it picks the next upgrade) *and* the
scale calibration (which is what the fixed ceiling consumes when it decides to stop).
**Falsifier:** rank correlation ≥ 0.9 and bounded scale error ⇒ the proxy is sound and §3.0's
hypothesis dies. **If it ranks correctly, say so plainly — that is a clean result too.**

**Transfer, per `ddm_bs2` (MAIN, 2026-08-24):** bs2 found that tri1's "property 2", read correctly,
means *"rate moves should land in the scorers' low-sensitivity subspace"* — which is `dD/dB` exactly,
so the born-small-basis route and this one are **bids on the identical quantity**, and R0's verdict on
whether the first-order surrogate is faithful **transfers to the basis route unchanged**. This arm
fires first only because its instrument is cheaper and already built.

### R1 — the A/B

**⚠ CARRY THIS IN EVERY R1 READING (MAIN, 2026-08-24). The renderer is the ONLY block that can host
this objective (§2.1), and it is 30,856 B = 72.80% of the 42,382 B demand. Therefore a PERFECT
`dD/dB` objective, fully realized, still caps at 72.80% of the demand and CANNOT close the gap
alone. It MUST compose.** Verified independently: `30,856 × 6.658590e-07 = 0.020546`;
`0.14821987563 − 0.020546 = 0.127674`, reproducing fb1. **No successful R0 or R1 is a route to
sub-0.12 on its own.** Under sy2's object-change law this is what makes the arm interesting rather
than weaker: a movable `r` **changes the object every closed cell was priced on**, and tri1's 28
pairs, 56 triples and fb1's bound were all priced at untrained `r`.

### R1a — measure the SEED-TO-SEED TRAINING floor (prerequisite for any KILL)

⚠ **CORRECTED 2026-08-24 — MAIN named the wrong noise first, and so did I by accepting it.** My
initial R1a said "repeat the `r` measurement on the same configuration." **That is vacuous:** the
instrument is deterministic — `pk4` measured repeat-noise **exactly 0.0** on the torch-CPU authority
("the negatives are signal"). A floor of 0.0 makes *every* difference resolvable, which is the mirror
of the 1.2× defect and just as wrong.

**The floor that actually governs the A/B is seed-to-seed TRAINING variance in `r`.** Two runs at the
same config with different seeds land on different weights and therefore different exchange ratios.
That is unmeasured on this vehicle.

**Design consequence, decided explicitly rather than inherited:** a two-arm ON/OFF A/B at one seed
each **cannot distinguish "the term moved `r`" from "seeds differ"**, so it is not a valid KILL
design. It remains valid for WIN (at ≥21.6× nobody cares about seed variance).

### ⚠ R1 RESTRUCTURED — the floor is the GATE on the treatment, not a preamble (MAIN, 2026-08-24)

**The defect:** "statistically indistinguishable" says nothing about what indistinguishability
*costs*. A measured floor is still a threshold, and a threshold whose price in score units is never
stated is the magnitude-dismissal defect one level deeper. Reproduced independently
(`unresolvable ΔS = credit·r₀·(1 − 1/f)`, credit 0.0205457 S, baseline damage `credit·r₀` = 0.4442 S):

| measured seed floor `f` | ΔS the experiment CANNOT resolve | **× remaining gap** |
|---:|---:|---:|
| 1.02× | 0.0087 | 0.31× |
| 1.05× | 0.0212 | **0.75×** |
| 1.10× | 0.0404 | **1.43×** |
| 1.20× | 0.0740 | **2.62×** |
| 1.30× | 0.1025 | 3.63× |
| 1.50× | 0.1481 | 5.25× |
| 2.00× | 0.2221 | 7.87× |

**A floor above ~1.05× already hides effects worth more than 0.7× the whole remaining gap.** And to
resolve an effect as small as R0's own measured mis-rank cost (0.001027 S) the two OFF seeds would
have to agree within **0.23%** — almost certainly unattainable.

**But power depends on EFFECT SIZE, and the two branches are not symmetric:**
- **WIN (`r < 1`, a 21.62× improvement) moves ΔS by 0.4237 = 15.0× the gap.** That is resolvable at
  *any* plausible floor. **The WIN branch survives this correction intact.**
- **KILL needs the floor.** A null result only means something if the design could have seen the
  effect.

**THE RESTRUCTURE, which makes R1 cheaper rather than more expensive:**

1. **Fire the two OFF seeds FIRST and STOP. ~3.4 h.** Deliver the measured floor **and, in the same
   sentence, the ΔS it renders unresolvable in gap units** from the table above. That is a complete,
   publishable result on its own: **it prices every future A/B on this trainer.**
2. **The floor GATES the ON arm.** If the floor is ≥1.2×, the third run cannot separate a 2.6×-of-gap
   effect from noise and **must not be fired as designed** — that needs seed averaging or a paired
   design, a different and larger experiment. Do not spend the 3.3 h to find out afterwards.
3. **Never emit KILL from an underpowered design.** If the floor is large the honest deliverable is
   **the floor itself** plus *"this design cannot answer the question at this power"*,
   `verdict_scope: INSTANCE`. That is a real finding, not a family verdict, and not a failure.

**PRICED, and affordable — so I am NOT returning a WIN-only experiment.** wd3 costs **93.23 s/epoch**
(median; mean 95.85, n=13 inter-checkpoint gaps; 101.8 min wall over 64 epochs) — **measured from the
F64 arm's own checkpoint mtimes**, this trainer, not borrowed. A 65-epoch run is **~101 min**.

| design | runs | est. wall |
|---|---:|---:|
| **2 OFF seeds (floor) + 1 ON arm** ← default | 3 | ~5.0 h (ON arm ~2× at k=2 ⇒ ~6.7 h) |
| 2 seeds per arm | 4 | ~6.7 h (~8.4 h with the ON 2×) |

MAIN's cheaper default is the right one: **establish the floor with two OFF seeds, then spend the
third run on the treatment.** The ON arm costs ~2× per step because the base rung plus one sampled
rung is two scorer passes (§6 cost note).

**Arms:** control = wd3 as-is (`ds1` inert, byte-identity asserted) · treatment = `mode="sampled"`,
`k=2`, uniform weights, `ceiling_multipliers` set from R0.
**Held fixed:** seed, schedule, starting checkpoint, teacher cache, all duals, `n`.
**Measured:** the allocation `choose_cheapest_passing_quantization` selects, its **exact** byte count,
and true `D` at that allocation through the frozen scorers.
**Primary quantity:** `r` on the renderer axis, control vs treatment.

### Bars — CORRECTED 2026-08-24 after a stop-hook catch

⚠ **My first version of this table was defective and I withdraw it.** It read
`KILL | r improves < 1.2× (within noise)`. That is a **magnitude dismissal**: a kill justified by a
small ratio, with no relative-significance number and no measured noise floor. MAIN caught it. The
correction below is not cosmetic — the original criterion would have manufactured a false family kill.

**Why 1.2× is not small.** Net score change from a shed is
`ΔS = ΔD − λ_B·ΔB = λ_B·ΔB·(r−1) = credit × (r−1)`. For the renderer block, `credit = 30,856 ×
6.658590e-07 = 0.0205457 S`, against a remaining gap of `0.028220`. **Credit is 0.73× the whole
gap**, so `r` is a violently high-leverage quantity. Recomputed independently from MAIN's table:

| improvement in `r` | `r` treated | net ΔS | Δ vs baseline | **× remaining gap** |
|---:|---:|---:|---:|---:|
| 1.05× | 20.590 | +0.4025 | 0.0212 | **0.75×** |
| **1.20×** | 18.017 | +0.3496 | **0.0740** | **2.62×** |
| 1.50× | 14.413 | +0.2756 | 0.1481 | 5.25× |
| 2.00× | 10.810 | +0.2016 | 0.2221 | 7.87× |
| 21.62× | 1.000 | 0.0000 | 0.4237 | 15.01× |

A 1.2× improvement moves net ΔS by **0.0740 — 2.62× the entire remaining gap.** Even 1.05× moves it
0.75× the gap. "Within noise" was false in the only currency that matters.

**And "within noise" was an unmeasured claim.** No one on this campaign has measured a noise floor on
`r`. Every "within noise" statement about the exchange ratio — MAIN's and mine — is currently
unfounded. **R1 therefore measures the floor first** (see R1a below); only a measured spread licenses
the word.

**The corrected bars. KILL and WIN are not two sides of one threshold — they are different questions.**

| Verdict | Condition | Meaning |
|---|---|---|
| **KILL** | `r` treated indistinguishable from baseline against the MEASURED seed floor (R1a) **AND the design is POWERED — i.e. the floor's unresolvable ΔS is smaller than the effect being denied, stated in gap units** | The mechanism did not resolve **on this instance**. `verdict_scope: INSTANCE`. **Never "the last route closes"** (#307). Requires ≥2 OFF seeds. **If the design is underpowered, KILL is UNAVAILABLE** — the deliverable is the floor plus "cannot answer at this power." |
| **POSITIVE** | any improvement **resolvable above the measured floor**, including 1.2× | Mechanism existence is established. Whether it reaches the bar is a **trajectory** question — does it keep going under more weight, does it compose — not a magnitude question. A cap-limited first result is not a family verdict. |
| **WIN** | `r < 1` at any rung | Break-even. Renderer bytes become purchasable. |

**The WIN bar is 21.62×, not 46.3×** (MAIN, 2026-08-24): the best exchange ratio ever measured on
this object is **tba1 D3 at 21.62×**, and since that counts the **seg leg only** it is a *lower*
bound on its own true ratio. W72's 46.3× is not the best — it is merely the one my charter handed me.
I withdraw the 46.3× framing.

**Improvement must come from MECHANISM, never magnitude.** dg2 measured `ratio ∝ B^−0.2748` with both
ends measured: shrinking a move *raises* its ratio. So no amount of taking smaller bites reaches
break-even. That is this term's entire thesis, stated as a number.

**Cost, MEASURED:** the base rung is always evaluated, so `sampled` with any `k` costs **2 scorer
passes per step ≈ 2× step time.** wd3 itself runs at **93.23 s/epoch** (median of 13 inter-checkpoint
gaps on the F64 arm; mean 95.85; 101.8 min wall / 64 epochs) — derived from that arm's own checkpoint
mtimes, so it is *this* trainer's number. I did not borrow nt1's 48.95 s/epoch, which is the **cl1**
trainer on MPS: a different object. Full sizing in R1a.

**Why ALIVE matters even though it does not reach sub-0.12:** tri1's 28 pairs and 56 triples, and
fb1's single-axis bound, were **all priced at untrained `r`.** A confirmed-movable `r` does not just
add a lever; it invalidates the price on every closed cell. That re-pricing, not the renderer's
30,856 B, is the real prize.

---

## 6b. R0 — FIRED AND MEASURED (2026-08-24). Zero new compute.

**Method:** R0 needed no scorer run at all. The trainer already retains, per arm, the full per-group
`quantization_sensitivity` table **and** a 4-way `quantization_race` with each candidate's measured
`hard_d_seg` / `d_pose`. R0 is a read of retained artifacts.
**Axis:** `[Darwin-mps frozen-scorer advisory]` · `score_claim=false` · `promotion_eligible=false`.
**Receipt:** `/Volumes/APDataStore/pact/ddm_ds1_cheap_to_shrink/R0/R0_RESULT.json`, 6,764 B,
sha256 `2a1a6ee1821cb0a38882b69ad9ea425ed9c4e79d08e80f4b5464870667332122`.
**Contention:** none consumed. `ddm_tv1` (282.8% CPU) and `ddm_df1` (204.8%) stayed untouched; I
waited on nothing and took no advisory slot.

**Dedupe first:** `W0_warm`'s four rows are **bit-identical** to `W0_reset`'s (same bytes, `d_seg`,
`d_pose`). They are ONE measurement, not two. Distinct arms = **3**, not 4 — the
"N negatives masquerading as convergence" genus, caught before counting.

### Finding 1 — the proxy misses its own pre-registered bar

| arm | stage | predicted order (worst→best) | TRUE order (worst→best) | ρ | τ |
|---|---|---|---|---:|---:|
| D56 | from_epoch_**0000** | u2 > u3 > u4 | u2 > **u4 > u3** | 0.50 | 0.33 |
| F64 | from_epoch_**0000** | u2 > u3 > u4 | **u3 > u4 > u2** | **−0.50** | −0.33 |
| W0 | from_epoch_**0060** | u2 > u3 > u4 | u2 > u3 > u4 | **1.00** | 1.00 |

**Pooled (n=9, 3 arms): ρ = 0.783, τ = 0.667.** F64 is a full **inversion**: the proxy calls uniform2
the worst allocation; truth calls it the **best**.

**⚠ The 0.90 bar is WITHDRAWN as inherited — MAIN's stop-hook, and MAIN is right.** "ρ ≥ 0.90" is the
conventional "good correlation" value. It is a **guessed default of exactly the same class as the 1.2×
I already withdrew**: a bar with no derivation and no cost attached. I inherited it in my own
pre-registration and should have derived it.

**The derived bar, from R0's own data.** The question a ranking bar must answer is *"what ranking
fidelity keeps the expected S loss from mis-ranking below X?"* — not *"is ρ conventionally good?"*
A mis-rank costs the **bytes it forgoes**. Measured, on the one arm where the mis-rank actually
forgoes bytes:

> **D56: `adaptive` 20,235 B vs `uniform4` 21,778 B ⇒ 1,543 B forgone = 0.001027 S = 3.64% of the
> remaining gap** — and `adaptive` was *better* on aggregate D there, so it is a pure loss.
> Across the n=3 arms only 1 of 3 forgoes bytes ⇒ **expected ≈ 514 B = 0.000342 S = 1.21% of gap per
> stage** (n=3, weak).

**The measurement is unchanged and still misses.** ρ = 0.783 with a full inversion on one of three
arms is not a fidelity anyone would accept; what changes is that the miss now carries a **price in
score units** rather than a convention. Report it as *"the proxy's unreliability costs ~1.21% of the
remaining gap per stage on this sample, and inverts outright on 1 of 3 arms"* — never as
*"ρ < 0.90."*

### Finding 2 — the proxy-designed allocation has never won

`adaptive` (built by `adaptive_allocation_from_sensitivity`) **fails the component gates in 3/3
distinct arms**, while being 1,211–3,770 B cheaper than the winner. The selected policy is
`uniform_int4_degenerate` in **4/4** arms. **The renderer's bytes are allocated by dumb uniform int4;
the entire sensitivity machinery is live, computed every stage, and never selected.** It is an
orphaned mechanism in production.

> ⚠ **CORRECTED by §6c.4 — do not read this row as evidence about allocation.** `uniform4` is the
> gate's own baseline and passes by identity (`x ≤ x`), so it **cannot lose**. The 4/4 record and the
> 0-for-3 record are both explained by the RULE. That the machinery is never selected is true as
> description; that it is *bad* does **not** follow, and I overstated it here.

**Honest nuance, against my own headline:** `adaptive` is not worse on *aggregate* D — in D56 it is
better AND cheaper (D 61.381 vs 64.572 at 20,235 vs 21,778 B) and in F64 it ties (50.638 vs 50.616).
It fails on **component** gates (`hard_cell`, `road_lane`), not on the composite. So the correct
statement is *"the gates reject it"*, not *"it is worse."* That distinction matters, and it surfaces
a separate live question for MAIN: the allocator is leaving **1,211–3,770 B** (0.0008–0.0025 S)
unbought per arm on component gates while the composite score would have accepted them.

### Finding 3 — my §3.0 hypothesis is CONTRADICTED, in the direction I predicted

I hypothesized the first-order proxy goes **blind at convergence** (`g → 0`). The data points the
**other way**: the two arms built at **birth** (`from_epoch_0000`) are the unreliable ones (ρ 0.50,
−0.50); the one built at **epoch 60** ranks **perfectly** (ρ 1.00). **n=1 converged arm, 3 points —
far too weak to assert the reverse law**, but it is enough that I must withdraw the hypothesis rather
than carry it. §3.0's Goodhart argument against *training on* the proxy stands on its own derivation;
its empirical prediction about *when* the proxy degrades does not.

**And §3.0's premise was already too strong.** `choose_cheapest_passing_quantization` refuses any row
with `measured is not True` (*"projected quantization row cannot select an allocation"*), so the proxy
is a **candidate generator**, never the selection authority — real measurement is. My "the allocator
may be systematically mis-ranking allocations today" was overstated: it is *overruled* every time and
falls back to uniform. That is a different, and more actionable, defect than the one I posited.

### What R0 changes for R1

- The rung ladder must **not** be built from `adaptive_allocation_from_sensitivity` alone — it is
  0-for-3. Build R1's `ceiling_multipliers` around the **uniform** ladder, which is what actually
  ships, and carry `adaptive` only as a probe.
- **Transfer to the basis route (bs2):** the surrogate is **not** faithful at the pooled level
  (ρ 0.783 < 0.90) and inverts on one arm, so the born-small-basis route **cannot** assume a
  first-order sensitivity surrogate will rank its candidate bases correctly either. That transfers
  unchanged, as MAIN asked.
- **Scope limit, stated plainly:** two of three arms were measured at **birth**, where `d_pose` is
  27–77 and `d_seg` 0.19–0.53 — a catastrophically untrained regime. R0 characterizes the proxy
  mostly *at birth*. A converged-state replication is owed and is the natural R0b.

---

## 6c. COMPONENT-GATE PROVENANCE — and a RETRACTION of my own nuance

**Receipt:** `.../ddm_ds1_cheap_to_shrink/R0/R0c_COMPONENT_GATE_PROVENANCE.json`, 2,400 B, sha256
`87ee5f84516f7d42…`. Retained artifacts only; zero new compute.

### ⚠ RETRACTION FIRST — the "unbought bytes" claim was mine and it was wrong

I wrote: *"the allocator is leaving 1,211–3,770 B unbought on component gates **while the composite
score would have accepted them**."* **I did not check that last clause, and it is false in 2 of 3
arms.** MAIN priced it at 8.90% of demand off my sentence. **The 8.90% is withdrawn — the number came
from W0_reset's 3,770 B, and W0_reset is the arm where `adaptive` is *catastrophically* worse.**

| arm | state | gate fires | composite net ΔS | composite verdict |
|---|---|---|---:|---|
| D56 | birth | `road_lane` only | **−3.191138** | would ACCEPT |
| F64 | birth | `hard_cell` only | **+0.020587** | would REFUSE |
| W0 | **converged** | **all three** | **+0.461468** | would REFUSE, emphatically |

**In 2 of 3 arms the gate refuses bytes the composite refuses too. Those bytes were never buyable.**
Only D56 shows the gate over-refusing, and its credit is **1,543 B = 0.001027 S = 3.64% of demand**,
not 8.90% — and D56 is a **birth**-state arm where `d_seg ≈ 0.41` and `d_pose ≈ 55`, so its −3.19 S
"improvement" is untrained-regime noise, not a shippable gain.

**On the only CONVERGED arm, `adaptive` is 4.49× worse on `d_seg`, 2.58× on `road_lane`, 1.09× on
`d_pose` — it loses on every axis simultaneously.** There is no reading of that as over-refusal.

### 1. Provenance: the thresholds are DERIVED — MAIN's fork resolves, and it closes

`ddm_wd3_scorer_aware_width_distillation.py:2014-2017`:

```python
"hard_cell_gate_pass": evaluation["hard_d_seg"]                 <= baseline["hard_d_seg"],
"road_lane_gate_pass": evaluation["cell_edges"]["road_lane_flips"] <= baseline["cell_edges"]["road_lane_flips"],
"pose_gate_pass":      evaluation["d_pose"]                     <= baseline["d_pose"],
```

with `baseline = evaluations["uniform4"]` (`:2007`). **There are no thresholds.** No hand-picked
constant, no borrowed number, nothing to put at class 4 and nothing to re-derive. This is a
**three-component Pareto-dominance test against an in-run measured reference** — value-provenance rung
**DERIVED-relative-to-measured-reference**. MAIN's "guessed ⇒ re-derive ⇒ bytes may be buyable" branch
is **FALSE**; the "derived ⇒ correctly refused ⇒ this closes" branch is the one that fires.

### 2. Which component fires: DIFFUSE, not a consistent Lane death

`road_lane` (D56) · `hard_cell` (F64) · all three (W0). **Not the same component each time**, so the
more interesting hypothesis MAIN offered — "`adaptive` always dies on Lane" — is **not supported**.
It is diffuse failure, which is the plainer and less exciting answer, and it is the true one.

### 3. Adversarial, as demanded — the gate is doing real work

I will not reach "too strict" by convenience, and the evidence does not support it. `road_lane_flips`
guards Lane specifically: **0.59% of area but 33.56% of model bits and the worst distortion class
(IoU 0.263)** per bl1. A composite is precisely the instrument that hides that, which is what `cb1`
measured when MyCar admitted while Lane rejected. In D56 — the one arm where the composite disagrees
— the gate fires on **`road_lane`, 1.278× more Lane flips**, which is exactly the damage a composite
launders. **The gate catching that is the gate working, not the gate malfunctioning.**

### 4. FIRST-CLASS RESULT — the selection is structurally rigged: the reference cannot lose

**This is the arm's most durable finding, and it outranks the byte figure it replaced.** Promoted from
a residual at MAIN's direction, 2026-08-24.

**`uniform4` passes its own gate by identity.** `baseline = evaluations["uniform4"]` (`:2007`), and the
gate row for `uniform4` evaluates `evaluation["hard_d_seg"] <= baseline["hard_d_seg"]` where
`evaluation` **is** `baseline`. So all three components are `x <= x` — **True by construction, on every
run, for all time.** `choose_cheapest_passing_quantization` then takes the cheapest passer, so
`uniform4` wins unless a rival strictly Pareto-dominates it on all three components at once.

> **The 4/4 `uniform_int4_degenerate` record is fully explained by the rule, with NO appeal to
> allocation quality whatsoever.**

**Consequence (a) — a whole class of future claim is uninformative by construction.** Any statement of
the form *"we raced allocations on this trainer and uniform won"* carries **zero evidence about
allocation** until the rule changes. The race has an entrant that cannot lose. **Nobody may re-derive
the 4/4 record as an allocation fact** — including me: my own §6b Finding 2 said the sensitivity
machinery is *"live, computed every stage, and never selected."* That remains true as a description of
the machinery, but it is **not** evidence that the machinery is bad. **The 0-for-3 record is evidence
about the RULE, not about allocation.** I overstated it and I correct it here.

**Consequence (b) — the frozen item is a DECISION RULE, not a constant.** Nothing here sits at class 4
as a *number*: there is no threshold to re-derive (§6c.1). What is frozen is (i) the **rule** —
Pareto-dominance on 3 components rather than net score — and (ii) the **choice of uniform-int4 as the
reference**. That is the operator's frozen-default class exactly: a choice made once, later read as a
constraint, and never revisited because it never presents as a number.

**I am not proposing to loosen it, and MAIN is not asking me to.** §6c.3 is why: the gate is doing real
work, and on the only converged arm the refused bytes were emphatically not buyable. This is recorded
so the record is not misread, not so the rule is relaxed.

**Cheapest decisive next step, NAMED and NOT FIRED** (MAIN's item 4): re-score the already-retained
`adaptive` candidate for **D56** — the only arm where the composite disagrees — **per class**, against
a properly derived per-class bar rather than dominance-vs-uniform4. It is a read of retained payloads,
zero new compute. **I do not recommend prioritizing it:** D56 is birth-state, the credit is 1,543 B
(3.64% of demand), and the converged arm points the other way. Logged so it is not lost, ranked low.

### R1a free-floor check — FIRED (minutes, zero compute). Answer: NOT free.

MAIN asked whether R1a might be another retained-artifact read, as R0 turned out to be. **It is not.**

**Method:** enumerated all **8** `COMPILED_CONFIG.json` and **11** `launch_manifest.json` under the
wd3 root and hashed each run's config-identity **with the seed excluded**, so any same-config pair
would collide.

- **Config-identity groups containing more than one run: ZERO.** Every compiled config is unique even
  ignoring the seed.
- **Distinct seeds across every wd3 run ever: `{20260815}` — exactly one.**
- `fire_rung2_w0warm` vs `..._v2` — the one genuine repeat-shaped pair — differ **only** in
  `expected_builder_sha256`. That is a relaunch after a *code* change at the **same seed**, so their
  spread would measure builder drift, not seed variance.
- No `--seed` variation embedded in any launch manifest.
- MAIN's warning was correct and I checked it: D56 / F64 / W0 are **different widths**, so their
  spread confounds width with seed and is **not** the floor. It was never a candidate.

**Verdict: the floor is NOT free. R1a fires as designed.** Receipt:
`.../R0/R1a_FREE_FLOOR_CHECK.json`, 1,001 B, sha256 `29a6cebda0b53914…`.

**And the stronger statement, which is the more useful one:** the seed has **never been varied on
this trainer at all**. Seed-to-seed variance on wd3 has not merely gone unmeasured — it has never
been *sampled once*. Every result this vehicle has produced sits at a single draw from a
distribution whose width nobody knows.

---

## 6d. R1 — SEALED, NOT FIRED. Launch order for MAIN.

**Status 2026-08-24: MAIN authorized R1a; it was NOT FIRED — see §6e, a pre-launch REFUSE. Stage 1 as
specified is impossible (seed change dies at resume; RNG is restored from the checkpoint regardless).
Stage 2 (R1b) remains SEALED and unfired.**

> ### ⚠ READ THIS FIRST IF YOU ARRIVED AT R1 ALONE
> **The allocator's 0-for-3 record is evidence about the RULE, not about allocation quality.**
> `uniform4` is the gate's own baseline and passes by identity (`x ≤ x`), so it **cannot lose** the
> race it defines (§6c.4). Any framing of the sensitivity machinery as an **"orphaned mechanism"** —
> including my own §6b Finding 2, and the version filed upstream as #1246 — is **qualified by this**:
> the thermometer is proven **UNCONSULTED**, not proven **BAD**. Nobody has measured whether a
> net-score rule would select it. Do not inherit the stronger claim; it is mine and it was wrong.

**STAGE 1 (fire this alone): `R1a` — two OFF seeds. ~3.4 h.**
- Arms: two control runs, `ds1` **inert** (`DEFAULT_CONFIG`), differing **only** in seed.
- Byte-identity of the inert lever asserted before launch (`test_inert_returns_the_same_object`).
- Ladder: built around the **uniform** allocation ladder, which is what actually ships.
  **NOT `adaptive_allocation_from_sensitivity` — it is 0-for-3** (§6b/§6c); carry it as a probe only.
- Deliverable: the measured seed floor `f` **and the ΔS it renders unresolvable, in gap units.**
- **STOP THERE.** Do not chain the treatment.

**STAGE 2 (fire only if Stage 1 licenses it): `R1b` — the ON arm. ~3.3 h.**
- `mode="sampled"`, `k=2`, uniform weights, `base_weight=1.0`, seed pinned.
- **GATE: fire only if the Stage-1 floor can resolve the effect being tested.** Floor ≥1.2× ⇒
  **do not fire as designed**; escalate to seed-averaging or a paired design, which is a different
  and larger experiment.
- WIN bar **`r < 1` = 21.62×** improvement (tba1 D3, seg-leg-only ⇒ a lower bound).
- **THE ASYMMETRY IS THE DESIGN, and a reader must not mistake an underpowered run for a failed
  one.** WIN moves ΔS by **0.4237 = 15.0× the gap** and is resolvable at *any* plausible floor.
  **Only KILL needs power.** So R1b is INFORMATIVE even under a bad floor — it simply cannot emit
  KILL. A large floor narrows which verdicts are available; it does not invalidate the run.

**Binding on both stages:** `verdict_scope: INSTANCE` · KILL unavailable if underpowered · the
**72.80% renderer cap** carried in every reading (a perfect objective on the only D-coupled block
still cannot close the gap alone — it MUST compose) · improvement must come from **mechanism**, never
magnitude (dg2: `ratio ∝ B^−0.2748`, so smaller bites are *worse*).

### The trainer's STANDING POWER LAW (carry forward; not an R1 detail)

For any A/B on wd3, the score effect a design **cannot** resolve at measured seed floor `f` is

> `unresolvable ΔS = credit · r₀ · (1 − 1/f)`,  credit = 30,856 B × 6.658590e-07 = **0.0205457 S**

so at `r₀ = 21.62` a floor of 1.05× already hides **0.75× the remaining gap**, and resolving an effect
as small as R0's measured mis-rank cost (0.001027 S) requires the seeds to agree within **0.23%**.
**This should have existed before any A/B on this trainer was ever designed.** It is a property of the
trainer and the operating point, not of this arm — every future wd3 A/B is priced by it.

**STILL OWED, and not to be dropped when R1 lands:**
- **R0b — converged replication. STILL OWED; does NOT drop when R1a lands.** 2 of 3 R0 arms are birth-state (`d_pose` 27–77). R0 characterizes
  the proxy mostly at birth; the one converged arm points *opposite* my withdrawn hypothesis.
- **The D56 per-class re-score** (§6c item 4) — named, zero-compute, ranked LOW.

---

## 6e. R1a — PRE-LAUNCH REFUSE. NOT FIRED. (2026-08-24)

MAIN verified the Metal slot free and authorized R1a. **I did not fire it.** Source inspection before
launch found R1a as specified is **doubly impossible**. Reported and stopped, not routed around.
Receipt: `.../R0/R1a_PRELAUNCH_REFUSE.json`, 2,435 B, sha256 `c20dfae6ca3ee5a6…`. **Cost of the
check: minutes. Cost avoided: 3.4 h.**

**Blocker 1 — a seed change is REFUSED at resume.** `:1131-1132`:

```python
if payload.get("config") != dict(expected_config):
    raise WD3Error("resume checkpoint config/source identity differs")
```

`expected_config=config` (`:2237`) is the **full compiled config, seed included**. Two configs
differing only in seed therefore differ, and the second arm **dies immediately on resume.**

**Blocker 2 — the seed would not have mattered anyway.** `:1139` `_restore_rng(payload["rng"],
generator)` restores **all four** RNG streams — `python`, `numpy`, `torch_cpu`, `generator` — from the
checkpoint. `seed_everything(config["seed"])` runs earlier and is then **overwritten**. Even with
Blocker 1 removed, both arms would be **bit-identical**: R1a would consume 3.4 h to measure zero.

### The deeper finding — the seed floor is the WRONG floor for this design

**For any A/B resuming from a COMMON warm checkpoint, this trainer is DETERMINISTIC by construction:**
both arms inherit identical randomness from the checkpoint, so the only difference between them is the
treatment. **The seed floor for the R1 A/B is structurally ZERO, modulo platform non-determinism.**

R1a was built to gate a confound that **the resume path already removes**. That is good news, not bad:
it means R1b's ON/OFF comparison is *exactly* interpretable, and it makes R1b **more** powerful, not
less. It also means my own §6d Stage-1 design was answering a question the design does not pose — a
second instance of the genus MAIN and I have now hit three times: a threshold or a floor carried
forward without checking whether it binds.

**Genuinely different seeds would require re-running `prepare_arm_birth` per arm**, which varies
**init** as well as trajectory — a *larger* floor than the A/B needs, since both A/B arms share a warm
start. Measuring it would answer a different question than the one gating R1b.

### Proposed substitute — NOT FIRED, needs MAIN's order

**`R1a′` — platform-determinism probe.** Two **bit-identical** re-runs: same config, same seed, same
checkpoint, **5 epochs** each. **~16 min total, versus 3.4 h.**
- **Bit-identical ⇒ the floor is ZERO**, R1b is exactly interpretable, and **KILL becomes available at
  full power** — the best available outcome.
- **Any divergence ⇒ a real floor exists**; measure it, extend the probe, and re-price R1b with the
  standing power law.

MAIN's order specified *"two OFF seeds"*; `R1a′` is a **different experiment**, so firing it would be
routing around the refuse. **It is proposed, not fired.**

### Determinism-by-construction is a POSITIVE property of the R1 design, not a caveat

**Reframed at MAIN's direction, 2026-08-24 — and MAIN's reframe is the correct one.** I had presented
the floor as a *tax on power*. On a warm-resume A/B it is the opposite:

> **Both arms inherit identical randomness from the checkpoint (`_restore_rng`, `:1139`/`:2213`
> restores all four streams), so THE ONLY DIFFERENCE BETWEEN THEM IS THE TREATMENT.**

A reader arriving here should find *"this comparison is exactly interpretable"* — **not** *"power
unknown."* There is no seed confound to subtract, because the resume path removes it by construction.
Any residual floor is platform non-determinism alone, which `R1a′` is designed to measure and which
is plausibly zero.

### R1a′ — ORDERED, NOT FIRED. Blocked on an operator gate. (2026-08-24)

MAIN ordered `R1a′` (two bit-identical 5-epoch re-runs). **It did not fire.** Two blockers, both
found before launch:

1. **`epochs` is inside `_birth_contract` (`:2098`)** and wd3 has **no** `stop-after-epoch`. A
   5-epoch run therefore needs a **fresh `prepare_arm_birth`** — the existing F64 birth is pinned at
   65. (Good news inside this: `output` is **not** in the birth contract, so two runs *can* share one
   birth and write to different dirs. The re-run design is sound; it just needs its own birth.)
2. **Minting that config requires `launch_authorized: true` plus two distinct claimed lanes**
   (`:496-502`, `raise` if `launch_authorized is not True`). **These are operator-gate fields.** Per
   my standing contract, a coordinating agent's order is not operator consent, so **I did not author
   them.** Reported instead of forged.

**To make `R1a′` fireable, MAIN supplies:** a compiled birth+train config pair at `epochs=5` carrying
`launch_authorized` and the two lane claim ids, into a fresh arm root. Then it is two governed
launches and a byte-compare — ~16 min, exactly as scoped. The experiment design is unchanged and
still correct.

---

## 7. PREDICTION ADJUDICATED

**FALSIFIED.** I predicted no `dD/dB`-shaped term had ever been measured on the dx2/hv1 lineage. The
falsifier fired twice: `quantization_sensitivity_table` (`wd3:848-880`) computes one in production
today, and §1.3 lists four measured shrink-damage ladders. I found the first myself while reading the
trainer; the recall agent found the second set. Either alone would have sufficed.

I note without excusing it that my falsifier was **well-specified and externally settled** — it named
"the corpus" rather than my own derivation — which is why it could fire against me at all. That was
the point of pinning it there.

**What survives, narrowed:** no `dD/dB` **objective term** has been built or measured on any vehicle
in this campaign. `nt1:231` self-grades it "unbuilt and unmeasured"; #242's flat-minima analogue was
proposed-only and on a retired vehicle. That is the gap this arm fills.

---

## 8. verdict_scope

**R1 carries `verdict_scope: INSTANCE`, fixed 2026-08-24.** One A/B, one vehicle, one ladder, one
seed policy. It cannot close a family and it cannot "close the last route" — that is the Catalog #307
paradigm-vs-implementation line, and my withdrawn KILL criterion violated it. A KILL from R1 means
*the mechanism did not resolve on this instance*, nothing wider.

**Scope: FORMULATION-level design + build, not a family verdict.** Nothing here is a measured result
about whether `dD/dB` is trainable. The build is unfired. §3's race is decided on *derivation* —
that we own an exact operator, so surrogates are mechanism reductions — not on measurement; a
measured race could still overturn it, and the module's injection seams make (a)/(d) cheap to swap in
if it does. §3.0's allocator-miscalibration claim is an explicit **hypothesis**, not a finding.
`r`-values in §6 are charter-supplied and were not independently re-derived.

---

## 9. NOT CLAIMED

- **R0 is a read of RETAINED artifacts, not a new measurement.** No Modal, no Metal burn, no scorer
  forward pass, no candidate archive. **Pointer UNMOVED.** No score claim, on any axis; R0 is
  `[Darwin-mps frozen-scorer advisory]` and inherits that axis from the artifacts it reads.
- **R0's 3 arms are n=3 with 3 allocations each — 9 points.** ρ=0.783 is a small-sample statistic and
  I do not claim a precise value for it, only that it sits below the pre-registered 0.90.
- **Two of three R0 arms are BIRTH-state.** R0 does not characterize the proxy at convergence; the
  one converged arm (n=1) points opposite my hypothesis and is too weak to establish the reverse.
- **R1 is NOT fired, and R1a was REFUSED at pre-launch (§6e), not attempted and failed.** No floor
  measured. No compute consumed. No KILL verdict available from this arm.
- **`R1a′` was ORDERED and still did NOT fire** — blocked on `launch_authorized`/lane-claim operator
  gate fields I must not author, plus a fresh birth (`epochs` ∈ birth contract). Nothing was run.
- **I did not measure platform non-determinism.** The claim that the trainer is deterministic given a
  checkpoint is read from the resume code (`_restore_rng` restores all four streams), NOT measured
  over 65 MPS epochs.
- **The seed floor is NOT free and R1a must actually run.** Checked: 8 configs, 11 manifests, one
  seed (20260815), zero same-config pairs.
- **I do not claim the sensitivity machinery is BAD.** §6c.4: its 0-for-3 record is evidence about the
  Pareto rule, not about allocation quality. My §6b Finding 2 overstated this and is corrected in place.
- **I did not test whether a net-score rule would select differently** — that would require re-scoring
  candidates under a different rule, which I named (§6c.4) and did not fire.
- **The lever is unfired and therefore has produced no evidence.** Binding-vs-inert is proven on
  closed-form tensors, not on the real model.
- **Not wired into wd3.** The call site is specified, not landed.
- **I did not measure `r`, `ΔD`, `ΔB`, or any allocation** on any real checkpoint.
- **I did not verify the charter's `r` values** (21.62×/46.3×/247.69×/349×/478.7×/687×/792×). I used
  them as supplied and flagged the W72 discrepancy rather than resolving it.
- **I published a defective KILL criterion and it stood until MAIN's stop-hook caught it.** §6 now
  withdraws it in full. It was a magnitude dismissal with no relative-significance number and no
  measured noise floor — the #404 class, in my own pre-registration.
- **No noise floor on `r` has been measured, by me or by anyone on this campaign.** Until R1a runs,
  no "within noise" claim about the exchange ratio is licensed from any arm.
- **I did not re-run WD4** to test whether its 1,590× is an unconverged-training artifact. Flagged only.
- **The 2× ON-arm figure is an operation count, not a timing.** wd3's 93.23 s/epoch IS measured (checkpoint mtimes), but I did not time the ds1-enabled step itself.
- **The 70,453 B HPAC model figure is nt1's**, from an `hv1_ep0634` parse-back — not dx2, and received
  not measured by me. The renderer's 30,856 B I did verify, by arithmetic against fb1.
- **The token-field edit axis is named, not analysed.** It is D-coupled and no trainer owns it; I
  scoped it out rather than treat it shallowly.

---

## STORES CONSULTED

`.omx/research/*.md` (full-corpus grep, 24 term families) · `.omx/state/canonical_task_status.jsonl`
+ 6 `.corrupt.*` snapshots · `.omx/state/graph_memory/nodes.jsonl` ·
`.omx/state/canonical_equations_registry.jsonl` · `.omx/state/subagent_progress.jsonl` ·
`.omx/research/harness_tasklist_bridge_20260803.jsonl` · `docs/` · `tools/` · `experiments/` ·
`src/` · PR130 intake code root (`hpac_self_compress.py`, `codec_hpac_integer.py`) ·
`ddm_nt1_…_20260824.md` · `ddm_tri1_…_20260824.md` · `ddm_fb1_…_20260823.md` ·
`ddm_jx1_…_20260822.md` · `ddm_sa1_…_20260818.md` · `ddm_wd4_…_20260821.md` ·
`ddm_keep01_…_20260818.md` · `ddm_tx1_…_20260819.md` §0 · `git log`.

**Payload retained:** `/Volumes/APDataStore/pact/ddm_ds1_cheap_to_shrink/`
(module 14,796 B `f58915f5891d1c53…` · tests 14,435 B `857232102460d57d…` · `RECEIPT.json` 852 B
`6581b2ef67b9a2be…`). Vertigo untouched — it is at 100%.

---

`dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]` — gap to 0.12 = 0.028220 ⇒ shed
42,382 B at fixed distortion, or 150 B at zero distortion.
