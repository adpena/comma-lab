# ddm_ds1 — the cheap-to-shrink objective: recall, derivation, mechanism race, build

**Date:** 2026-08-24 · **Arm:** ddm_ds1 · **Pointer:** UNMOVED · **No run fired.**
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
**Falsifier:** rank correlation ≥ 0.9 and bounded scale error ⇒ proxy is sound, §3.0's hypothesis dies.

### R1 — the A/B

**Arms:** control = wd3 as-is (`ds1` inert, byte-identity asserted) · treatment = `mode="sampled"`,
`k=2`, uniform weights, `ceiling_multipliers` set from R0.
**Held fixed:** seed, schedule, starting checkpoint, teacher cache, all duals, `n`.
**Measured:** the allocation `choose_cheapest_passing_quantization` selects, its **exact** byte count,
and true `D` at that allocation through the frozen scorers.
**Primary quantity:** `r` on the renderer axis, control vs treatment.

### Bars — stated honestly

The best measured renderer rung is **W72 at r ≈ 46.3×** (charter-supplied; `fb1:47` states the same
rung as `−10,879 B ⇒ seg ×116.8` — **these are different framings and I did not reconcile them; do
that at consumption**). Break-even needs `r < 1`. **So full success asks for a ~46× improvement from
one regularizer. I do not predict that, and I will not pre-register a bar I expect to miss as if I
expect to hit it.**

| Verdict | Condition | Meaning |
|---|---|---|
| **KILL** | `r` improves < 1.2× (within noise) | `dD/dB` is not trainable on this vehicle. The last route closes. **Decisive for the campaign** — say it plainly. |
| **ALIVE** | `r` improves ≥ 2× | The derivative moves. Measure `dr/d(rung weight)` to extrapolate the reachable ceiling. |
| **WIN** | `r < 1` at any rung | Renderer bytes become purchasable. |

**Cost, honestly:** the base rung is always evaluated, so `sampled` with any `k` costs **2 scorer
passes per step ≈ 2× step time.** I have **not** measured wd3's s/epoch and I am not borrowing
nt1's 48.95 s/epoch, which is the *cl1* trainer on MPS — a different object. **MAIN should pin
wd3's epoch cost before sizing the run.**

**Why ALIVE matters even though it does not reach sub-0.12:** tri1's 28 pairs and 56 triples, and
fb1's single-axis bound, were **all priced at untrained `r`.** A confirmed-movable `r` does not just
add a lever; it invalidates the price on every closed cell. That re-pricing, not the renderer's
30,856 B, is the real prize.

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

**Scope: FORMULATION-level design + build, not a family verdict.** Nothing here is a measured result
about whether `dD/dB` is trainable. The build is unfired. §3's race is decided on *derivation* —
that we own an exact operator, so surrogates are mechanism reductions — not on measurement; a
measured race could still overturn it, and the module's injection seams make (a)/(d) cheap to swap in
if it does. §3.0's allocator-miscalibration claim is an explicit **hypothesis**, not a finding.
`r`-values in §6 are charter-supplied and were not independently re-derived.

---

## 9. NOT CLAIMED

- **No run fired.** No Modal, no Metal burn, no candidate archive, no scorer forward pass. **Pointer
  UNMOVED.** No score claim of any kind, on any axis.
- **The lever is unfired and therefore has produced no evidence.** Binding-vs-inert is proven on
  closed-form tensors, not on the real model.
- **Not wired into wd3.** The call site is specified, not landed.
- **I did not measure `r`, `ΔD`, `ΔB`, or any allocation** on any real checkpoint.
- **I did not verify the charter's `r` values** (21.62×/46.3×/247.69×/349×/478.7×/687×/792×). I used
  them as supplied and flagged the W72 discrepancy rather than resolving it.
- **I did not re-run WD4** to test whether its 1,590× is an unconverged-training artifact. Flagged only.
- **I did not measure wd3's epoch cost.** The 2× figure is an operation count, not a timing.
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
