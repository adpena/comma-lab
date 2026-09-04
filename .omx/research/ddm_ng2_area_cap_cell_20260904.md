# DDM NG2 — the area-cap cell: closing the other side of a one-sided constraint set

**Date:** 2026-09-04
**Arm:** `ddm_ng2_area_cap_cell`
**Axis:** `[seal + bounded macOS-CPU mechanism smoke only; no Metal, no Modal, no contest eval]`
**Disposition:** **SEALED / BOUNDED-SMOKE-PASS / BURN-NOT-FIRED — MAIN fires.**

## Result first

The cell is sealed and the lever is exactly one config block. Three things I had to correct on the
way, and each makes the race sharper rather than weaker.

1. **The law's own tolerance does not transfer.** `chan_vese_area_constraint_birth_balance_v1`
   defaults to `delta = 0.25`, i.e. an equilibrium at 1.25× GT area. That constant was derived on
   the v7.5 witness vehicle, whose rare classes ran away to **13.8× / 4.6× GT**. On the born
   vehicle the excursion `ddm_sd1` measured peaks at **1.0929× / 1.0580×** — the v7.5 equilibrium
   sits *above every ratio this object ever reaches*, so carrying `delta = 0.25` would have shipped
   an **inert** cap that looked exactly like a fired lever. The FORM transfers; the constant is
   re-derived at this vehicle's operating point (`[[m21]]` constants→laws, `[[m143]]` cross-regime
   transfer).

2. **The soft area is the wrong quantity here, by 2.43×.** The law's differentiable stand-in for
   the region area `A_c = ∫H(φ_c)` is the softmax partition mass. MEASURED at this cell's exact
   start state, softmax(T=1) mass over-states **Lane by 2.4336×** and **Movable by 1.2823×** while
   the realized argmax area is only **1.0407× / 1.0244×** of GT. A soft-only hinge would therefore
   have retracted hard from update one against a quantity that is 2.4× the thing being capped. The
   cell ships a **straight-through** area instead: the VALUE is the exact argmax area, the GRADIENT
   is the softmax Jacobian `softmax_c(1−softmax_c)` — which is precisely the discrete `δ(φ_c)`
   boundary measure the law's variational gradient asks for. The temperature is the scorer's own
   **T = 1, frozen**, never the annealed `tau`, because `ddm_sd1` measured that a moving temperature
   deflates a frozen field's reported quantity by −40.54%; an area cap read through `tau` would
   inherit that defect wholesale.

3. **`verify_inputs()` has been refusing in the working tree since 4a7ae5ca0.** ng1 recorded the
   packet-schema drift and inherited pins to work around it. MAIN's 4a7ae5ca0 cured that drift by
   editing `PINNED_SHA256["packet_schema"]` *inside the trainer* — which moved the trainer's own
   bytes without moving the burn prep's pin OF the trainer, so `qbt.verify_pins()` passed while
   `qbr1.verify_inputs()` raised `pinned input drifted: qbt_trainer` one link up. ng2 had to re-pin
   the trainer anyway (a new loss term changes its bytes), so this landing closes that link too.
   The charter's premise "the working tree compiles again" was half true; it is fully true now.

The cap is **binding, not decorative**: at the control's measured step-2,000 peak the retraction
force dominates the birth force by **2.283× (Lane) / 2.377× (Movable)**.

## Verified at source (every premise carries `path:line`)

| claim | evidence | label |
|---|---|---|
| the live constraint set is RECALL-ONLY dual ascent on Lane/Movable; nothing caps area | `experiments/ddm_qbt1_qbflow_trainer.py:614-640` `dual_ascent_margin_constraints`; bounds Lane 0.12 / Movable 0.009 at `:183-184` | MEASURED |
| the cell's objective is the burn prep's `fairform_objective`, NOT `qbt.joint_objective` | `experiments/ddm_qbr1_born_fairform_burn_prep.py:354-420`, called at `:635` inside `run_config` | MEASURED |
| the per-class growth pressure is `100 * lambda_c^dual * per_class_expected_flip(c)` | `ddm_qbr1_born_fairform_burn_prep.py:389-392` | MEASURED |
| `A_GT_c` comes from the same bincount the balanced class weights use | `ddm_qbt1_qbflow_trainer.py:707-721` `derive_balanced_class_weights`; `:724-739` `selection_gt_area_fractions` | MEASURED |
| the registered law's callable, consumed not re-implemented | `src/tac/canonical_equations/chan_vese_area_constraint_birth_balance_20260708.py:157-176` `area_constraint_lambda` | MEASURED |
| the law's own `delta` default is 0.25 (equilibrium 1.25× GT) and its anchor's runaway was 13.8×/4.6× | same module `:152` `DEFAULT_AREA_TOLERANCE`; `:10-16` measured ep125 table | MEASURED |
| the area cap ships as `one_sided_area_cap_penalty` + `realized_class_area_ste` + `derive_area_cap_lambdas` | `ddm_qbt1_qbflow_trainer.py:811-864`, `:781-808`, `:742-778` | MEASURED (this arm) |
| the DSL twin forbids a global batch-area hinge (per-pair before the batch mean) | `src/tac/witness_dsl/curriculum_dsl.py:4839-4841` (`AreaConstraintBirth` docstring) | MEASURED |
| the burn prep's `verify_inputs()` refused in the working tree before this landing | `ddm_qbr1_born_fairform_burn_prep.py:70-82` `EXPECTED_SHA256["qbt_trainer"]` vs the worktree trainer sha; reproduced (`QBR1Error: pinned input drifted: qbt_trainer`) | MEASURED |
| `config_identity` ignores `action/output/resume_from/launch_authorized/scorer_lane/metal_lane` — `area_cap` is therefore part of the identity | `ddm_qbr1_born_fairform_burn_prep.py:156-165` | MEASURED |
| `storage_preflight` refuses any output outside four AP custody roots | `ddm_qbt1_qbflow_trainer.py:326-351` | MEASURED |
| the cold control of record (six milestones, recomputed from components) | ng1 memo `.omx/research/ddm_ng1_warm_transition_burn_design_20260904.md`, read back from `runs/seed_20260902/control_native100/milestones/step_*/MILESTONE.json` | TRANSFERRED (ng1, same cell, read-only) |
| the excursion is rare-class over-paint peaking at step 2,000, mass-conserving ±7.550e-04 | `.omx/research/ddm_sd1_surrogate_exact_decoupling_20260904.md` §4 | TRANSFERRED (sd1, same cell) |
| the τ-schedule leg is −40.54% on a frozen field; at τ=0.05 the surrogate peaks with the exact term | sd1 §1 | TRANSFERRED (sd1, same cell) |
| vr1 row 3 is the Chan-Vese one-sided area cap and names this exact plug point | `.omx/research/ddm_vr1_v7_v11_signal_recall_20260903.md:139` | TRANSFERRED |

## The derivation, with its receipts

### A_GT_c — the bincount (MEASURED, this arm)

`selection_gt_area_fractions` over the sealed 32-pair selection, the trainer's own target read
(PyAV `gt_n600.npz`, 6,291,456 pixels):

| class | count | A_GT_c |
|---|---:|---:|
| Road | 1,454,559 | 0.231195926666 |
| **Lane** | **37,506** | **0.005961418152** |
| Undrivable | 3,119,478 | 0.495827674866 |
| **Movable** | **78,368** | **0.012456258138** |
| MyCar | 1,601,545 | 0.254558722178 |

Cross-check: `w_c · 5 · A_GT_c = 1` for every class against `derive_balanced_class_weights` — the
two functions read one bincount. No pair in the selection is missing either capped class (Lane per-pair
min 0.00307, Movable min 0.00151), so the per-pair hinge is well posed everywhere.

### F_c — the birth force (MEASURED, this arm, on the control's own history)

The law's `F_birth` is a birth loss WEIGHT. Its analogue here is the dual's effective class weight
`100·λ_c^dual`, read from `runs/seed_20260902/control_native100/history.jsonl` over the excursion
window (updates 1–2,000):

| class | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|
| Lane | **0.6793084080313092** | 0.4971915 | 0.9537405 | 0.0071682 | 4.9977490 |
| Movable | **2.3063736731** | 1.9990407 | 2.7437428 | 0.0077576 | 6.0985959 |

**The honest caveat, stated because it bounds the equilibrium claim:** `F_c` is not a constant on
this vehicle. The dual starts at 0, spikes to 4–5 within ten updates, then settles into a narrow
band (p10–p90 spread 1.92× for Lane, 1.37× for Movable). The median is the robust central value;
the spread travels with it. Consequently the equilibrium `A* = (1+δ)A_GT` is a **nominal target**,
and the binding arbiter is falsifier 2 (the MEASURED area at step 2,000), never the formula.

### δ_c — the tolerance (MEASURED, this arm, at the exact start state)

One CPU forward of both 16-pair chunks from the sealed same-start r10 EMA shadow, HT-weighted:

| class | GT area | realized argmax area | **argmax/GT** | softmax(T=1) mass | softmax/GT |
|---|---:|---:|---:|---:|---:|
| Lane | 0.0059392293 | 0.0061809540 | **1.0406996660** | 0.0144534550 | **2.4335573** |
| Movable | 0.0143447876 | 0.0146948497 | **1.0244034323** | 0.0183940397 | **1.2822804** |

Two readings travel with this table.

* **δ_c = argmax/GT − 1**: `δ_Lane = 0.04069966601010533`, `δ_Movable = 0.02440343225897945`.
  The equilibrium is placed at the area each class ALREADY occupies at update zero, so the
  constraint says exactly one thing for both classes — *do not grow past where you began* — and
  its step-0 retraction is by construction its own equilibrium force rather than an arbitrary
  extra push. This is why `delta` is per-class here and shared in the v7.5 twin: the two classes
  start 1.67× apart in excess and their birth forces differ 3.4×, so a shared knob would impose a
  DIFFERENT effective constraint on each class. The per-class form is the uniform statement; the
  shared knob would have been the arbitrary one.
* **soft/GT vs argmax/GT** is the measurement that killed the soft-only formulation (Result 2).
  Cross-check on the axis sd1 used: sd1's DALI-authority read gives step-0 ratios 1.03339 / 1.02591
  against my PyAV 1.04070 / 1.02440 — consistent with the known 1.011–1.017× DALI/PyAV `d_seg`
  offset on this n32 selection. The cell uses the PyAV numbers because PyAV is the target the loss
  actually sees; using DALI areas inside a PyAV-targeted loss would be a second lever.

### λ_c — the stiffness (DERIVED through the registered callable)

`λ_c = F_c / (δ_c · A_GT_c)`, evaluated by
`tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708:area_constraint_lambda`
and re-derived by the config validator at every run start (a hand-edited stiffness that its stated
inputs do not imply is refused):

| class | λ_c | equilibrium ratio | dominance at the control's step-2,000 peak |
|---|---:|---:|---:|
| Lane | **2799.797098192266** | 1.0406996660 | **2.2828197159** |
| Movable | **7587.368191653098** | 1.0244034323 | **2.3771246349** |

Dominance = `(A_peak/A_GT − 1)/δ_c` via the law's own `dominance_at_runaway`, using sd1's measured
peak ratios 1.09291 / 1.05801. Both above 1, so the retraction genuinely dominates the birth force
where the excursion actually is — the operator's v7.5-era requirement, re-checked at this vehicle's
much smaller runaway.

### The form that ships

```
L_area = Σ_{c ∈ {Lane, Movable}}  (λ_c / 2) · Σ_b w_b · relu(A_c,b − A_GT_c,b)²  /  Σ_b w_b
A_c,b  = argmax-area value + softmax(T=1) Jacobian gradient   (straight-through)
```

One-sided at `A_GT` (nucleation and recall stay unopposed below GT); **per-pair** hinge against that
pair's own GT area before the HT mean (a pair holding little of a class must not be able to paint up
to the selection mean for free); consumes the SAME realized through-R logits the expected-flip terms
already have, so there is **no extra scorer forward**.

## The telemetry row (free, default ON)

`seg_expected_flip_realized_tau_ref` — the same surrogate at the fixed reference `τ_ref = 0.05` —
is emitted beside the annealed value in `history.jsonl` at **every** update, under `torch.no_grad()`
on detached logits. It touches no parameter, consumes no RNG, and defaults ON per the "observability
is not gate-able when score-neutral" law. sd1's finding is what it cures: under the anneal the
reported loss fell monotonically −35.1% across the whole run while the exact argmax rose to a peak at
step 2,000 and ended +9.56% above its start, because the schedule leg (−40.54% on a frozen field) is
8.4× the field's own signal (+4.85%). At `τ = 0.05` the surrogate has the correct sign in 5 of 5
windows and peaks at the same milestone as the exact term. This row makes ng1's falsifier 2 readable
in real time instead of post hoc.

## Pre-registered falsifiers (fixed before the burn)

1. **PRIMARY — the cap must act on the mechanism.**
   `S_hat(5,000) < 0.42514878445269977` **AND** `S_hat(2,000) < 0.48567677825279465` (the cold
   control's endpoint and its over-paint peak).
   *If it fails:* the excursion is not rare-class over-paint on this vehicle and vr1 row 3 is
   refuted at FORMULATION scope for the born object — not the family.
2. **The cap must actually bind.** Realized argmax area / GT area at step 2,000 within each class's
   own measured start ratio (**Lane ≤ 1.04070, Movable ≤ 1.02440**; the charter's rounded form is
   1.03 for both).
   *If it fails:* λ_c is too soft. The FORM is untouched and the SCALE is the next single lever —
   the law's own scale field is `ASSUMED_AWAITING_VERIFICATION`, so this outcome is what pays it.
3. **The fixed-τ telemetry must be faithful in-loop.** `seg_expected_flip_realized_tau_ref` must
   peak at the same milestone as `d_seg_hat`.
   *If it fails:* sd1's fixed-τ faithfulness does not survive inside the loop; the telemetry row is
   wrong, not the lever.

Read the DECOMPOSITION at every milestone, never the composite: the control's damage is 91.20%
d_seg, and a cap that "fixed" S_hat by moving bytes or pose would be a different finding.

## The single lever, proven

The cap cell and its matched control are BOTH compiled fresh through `qbr1.compile_cell` in this
tree, so the only thing separating them is the lever. The sealed control on disk is deliberately
**not** deep-copied (ng1's method): its pins name the previous trainer, which by construction cannot
carry a new loss term, so inheriting them would seal a config no tree can satisfy.

```
differing keys: ["area_cap", "cell_id", "output"]      allowed: the same three
```

Held identical and asserted: `objective` (native_interface_weight 100), `ema`, `schedule`,
`initial_state`, `learning_rate` 2e-4, `margin_constraints`, both τ endpoints, `pair_ids`,
`selection_weights`, `total_steps` 5,000, `milestones`, `seed` 20260902, `resume_from` = null
(**COLD** transition — the same transition the control took, so the pair isolates the cap and never
composes it with ng1's warm lever; `[[m164]]` union ≠ sum of legs).

### The pin delta (this is NOT a same-pins twin, and it cannot be)

| pin | sealed control (`106d0dd0…`) | ng2 (`35e4d678e…`) |
|---|---|---|
| `qbt_trainer` | `6eda9c20…` | `9f74641c…` |
| `qbt_packet_schema` | `5405ccd4…` | `7fe5285f6…` |

The trainer pin moves because it carries the lever. The packet-schema pin moves because of MAIN's
4a7ae5ca0 (eq1's addendum was append-only; the schema body is unchanged) — the sealed QBR1 tree keeps
the old pin and is untouched. Everything else is byte-identical. `uncommitted_diffstat_at_seal` is
empty: the seal was taken at a clean committed revision.

## Bounded CPU smoke — PASS, and one number argues against the cell

The Metal lane is held by the live QBR1 chain (`pgrep -f ddm_qbr1_cell_chain` → 95296/95299/95317
throughout), so every segment ran on **CPU**; `_run_resume_smoke_segment` forces `device="cpu"`
itself. This arm did **not** touch `.omx/state/active_lane_dispatch_claims.md` — the chain reads it
every poll and a malformed edit raises `CLAIMS_UNREADABLE` and refuses an 18-hour burn. 0 Metal /
0 Modal / 0 contest-eval invocations. Two real B=16 first updates from the identical sealed start.

| check | result |
|---|---|
| **NO-OP DETECTOR** — cap step-1 vs control step-1 live state | **DIFFERENT** (`8819e0b7e2c8c9f1…` vs `27f514180db2b4cd…`) |
| — and their re-encoded archives differ too | yes |
| **TELEMETRY IS SCORE-NEUTRAL** — control step-1 vs ng1's PRE-telemetry cold reference | **BIT-IDENTICAL** (`27f514180db2b4cda57289bbeb4be5ca8daf64e874921c92ba5c08d613c30973`) |
| first-update displacement `‖θ₁−θ₀‖₂`, control | 0.05588674077623233 (ng1 measured 0.055886740188786026) |
| first-update displacement, area-cap arm | 0.055886117042189765 |
| **DIFFERENTIAL** — cap penalty and gradient when both classes sit under GT | **exactly 0.0** and **exactly 0.0** |
| — same helper, same λ, both classes over GT | **281.25 > 0** |

The telemetry row is the strongest receipt here and it is not an assertion: **ng1 ran this exact
one-update cold segment before the row existed, and the trained state after the row was added is
bit-identical.** A score-neutral observability row that defaults ON has been *measured* to change no
trained byte. (The displacement scalars differ at the 11th decimal — 1.05e-11 — from float summation
order in this arm's helper; the state hashes are equal, so the weights themselves are identical.)

The smoke ran **twice**, and the second run reproduced every hash, every displacement, every cap
energy and every telemetry value to the last digit — a free determinism receipt on top of the
mechanism one. Only the logit-space gradient norms moved, in the 9th significant figure (CPU thread
reduction order), which is why the parity multiplier below is quoted as a range. Wall clock 74.5 s
and 76.2 s; retained payload 67 MB per arm.

**A hazard for whoever re-runs this:** the probe's process high-water mark is **41.46 GiB**
(`peak_rss_bytes`), because two real B=16 training updates and the gradient probe share one address
space. It completed with ~43 GB available while the Metal chain was running, but not with much room.
Budget it, or run the segments in separate processes.

### The gradient-scale receipt — and it argues against my own cell

The balance law is stated in AREA units. The realized effect on the weights travels through
LOGIT-space gradients, whose normalizations differ (the recall term divides by the class's own pixel
count, the area term by the frame). So I measured it rather than assuming it. At the exact start
state, chunk 0, τ = 0.15, with the derived λ:

| term | ‖∂·/∂logits‖ | loss value |
|---|---:|---:|
| `100 · realized expected-flip` | 0.0119239371 | 0.5018186569 |
| recall dual penalty (Lane + Movable) | 0.0069750678 | 0.1805762053 |
| **the area cap** | **8.7164408e-05** | 0.0008220663 |

**cap / recall = 0.0124966 · cap / realized = 0.0073100.** In logit space the cap is **~1.25% of
the recall term it is meant to counterbalance.** Its force is linear in the overshoot, so at the
control's step-2,000 peak it would be ~2.3× larger — still about 3% of recall. **This lowers my own
prior that falsifier 1 fires, and I am recording it before the burn rather than after.** The
multiplier that would put the cap at gradient-norm parity is **×80.02–80.05** across the two smoke
runs, recorded as `lambda_scale_for_recall_gradient_parity`.

I did **not** re-scale λ. Tuning the stiffness to a step-0 logit-space proxy would trade the derived
balance for a fitted constant and would make falsifier 2 unfalsifiable — the cap would bind by
construction. The derived λ is the pre-registered position; the MEASURED area at step 2,000 is the
arbiter; and if falsifier 2 says the cap is too soft, the scale is the next single lever with its
multiplier already in hand. That is exactly the `ASSUMED_AWAITING_VERIFICATION` the law's own second
anchor has been owed since 2026-07-08.

### What the cap actually did at update 1 (from the run's own `history.jsonl`)

`area_cap_over_Lane` 3.5655e-04 and `area_cap_over_Movable` 3.0492e-04, so the realized retraction
force `λ_c · over_c` is **0.998 (Lane)** and **2.314 (Movable)** against equilibrium forces of 0.679
and 2.306 — Movable is within **0.3%** of its own equilibrium at the first update, and Lane sits
1.47× above its because this chunk's Lane area runs above the selection HT mean. The mechanism does
what the derivation says it does.

The telemetry row on the same update: `seg_expected_flip_realized` **0.005018184892833233** at
τ = 0.15 beside `seg_expected_flip_realized_tau_ref` **0.0029314844869077206** at τ = 0.05 — the
same field, 41.6% apart, which is the whole point of the row.

## MAIN fire command

Preconditions MAIN owns: the QBR1 chain has released the Metal, a live scorer claim and a live Metal
claim exist, and the sealed tree is unchanged. MAIN copies the sealed config to
`authorized_configs/`, sets `launch_authorized: true`, and binds both claim IDs — all three fields
are in `config_identity`'s ignored set, so binding them does not disturb the sealed identity.

```bash
SRC=/Volumes/VertigoDataTier/pact/ddm_ng2_area_cap/sealed_source_54161c2800
$SRC/.venv/bin/python $SRC/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng2_area_cap/launch/seed_20260902_area_cap_control_native100 \
  --cwd $SRC \
  --purpose "NG2 area-cap cell seed_20260902_area_cap_control_native100" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib 2.3959503173828125 \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt DONE.json \
  -- $SRC/.venv/bin/python $SRC/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config /Volumes/APDataStore/pact/ddm_ng2_area_cap/authorized_configs/seed_20260902_area_cap_control_native100.json
```

**Fire from the SEALED tree.** Its `verify_inputs()` was run inside it as part of the snapshot and
returned PASS on all four pinned inputs; its `upstream/` and `experiments/results/mlx_fleet_gt_cache`
are symlinks to the repo (the same arrangement `sealed_source_106d0dd0_v2` uses), because
`git archive` carries only tracked files and those two are the large sha-pinned inputs.

Cost: one cell, **~2.95 h** measured on the identical object, **~1.375 GB** retained. No control
re-burn — the control is the already-measured seed-20260902 row.

**A scheduling correction MAIN should have:** the charter assumed the chain releases the Metal
around 14:00Z. At seal time the chain is on **cell 3 of 6** (`seed_20260903/control_native100`, past
its step-2,000 milestone), so roughly **3.5 cells ≈ 10.3 h** of chain remain before ng1's warm cell,
let alone this one. APDataStore holds **23 GiB free** against an 8 GiB reserve; four remaining chain
cells plus ng1 plus ng2 project ≈ 8.25 GB, which clears the reserve but not by much.

## Custody (ALWAYS KEEP THE PAYLOAD)

| artifact | path |
|---|---|
| derivation receipt (F_c, δ_c, λ_c, all inputs) | `/Volumes/APDataStore/pact/ddm_ng2_area_cap/DERIVATION.json` |
| seal receipt (single-lever diff, pin delta, sealed-source binding, falsifiers) | `…/ddm_ng2_area_cap/SEAL_RECEIPT.json` |
| **sealed cap-cell config** | `…/ddm_ng2_area_cap/sealed_configs/seed_20260902_area_cap_control_native100.json` |
| matched control (reference recompile, marked do-not-fire) | `…/sealed_configs/matched_control_of_record_seed_20260902_control_native100.reference.json` |
| sealed source manifest | `…/ddm_ng2_area_cap/SEALED_SOURCE_MANIFEST.json` |
| **sealed source tree** (rev `54161c2800…`) | `/Volumes/VertigoDataTier/pact/ddm_ng2_area_cap/sealed_source_54161c2800/` |
| bounded smoke result + both arms' retained payloads | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng2_area_cap/bounded_smoke/` |
| run output root (empty; MAIN's cell writes here) | `…/ddm_qbr1_born_fairform_burn_prep/ng2_area_cap/runs/seed_20260902_area_cap_control_native100` |
| code + 31 tests | `experiments/ddm_ng2_area_cap_cell.py`, `src/tac/tests/test_ddm_ng2_area_cap_cell.py` |

`authorized_configs/` is **not** written by this arm. Nothing was written under the live chain's
`runs/`, `authorized_configs/` or `CHAIN_LEDGER.jsonl`; the claims ledger was not touched; the
seed-20260902 control run was opened read-only.

## Equations leg (`tac.canonical_equations`)

**`chan_vese_area_constraint_birth_balance_v1` — CONSUMED (form) and REFINED (domain).** The
trainer delegates every stiffness to the module's own `area_constraint_lambda`, the seal validator
re-derives each λ from the config's stated inputs, and the numpy reference `area_penalty` is
asserted against the torch term in the test suite.

A `domain_refined` event is appended (`event_type: domain_refined`, agent `ddm_ng2_area_cap_cell`,
2026-09-04T01:29:06Z). It records ONE inclusion — the FORM and the balance law transfer to the
qbf1-born vehicle with a straight-through area and a bincount-derived `A_GT_c` — and THREE
exclusions, each MEASURED by this arm:

* `DEFAULT_AREA_TOLERANCE = 0.25` and its derived v7.5 literals (λ_lane 683.8 / λ_movable 322.6) on
  any vehicle whose runaway is not the ep125 13.8×/4.6× regime — here the runaway peaks at
  1.0929×/1.0580×, so a 1.25× equilibrium is inert;
* the softmax-partition-mass area estimator on thin classes at this logit scale (2.4336× / 1.2823×
  bias against a 1.0407× / 1.0244× truth);
* reading the area through the annealed `tau` (sd1's −40.54% schedule leg).

**No performance anchor is appended.** Whether the cap LOWERS d_seg is unmeasured until the cell
burns; anchoring the law on a design would be the thing this campaign keeps extincting. The law's
`domain_of_validity.vehicle` field is untouched (`softmax_of_sdf_levelset_witness`).

`ema_decay_run_geometry_v1` is consumed unchanged and IN-DOMAIN: the cell inherits the control's
sealed decay 0.9990793899844618 and the strict `check_ema_executable_law_matches_sealed_law` gate
sees no change.

## Scope and limits (these travel with the numbers)

* **Axis.** Every S_hat quoted is `[macOS-MPS n32 stratified advisory]`; the derivation and smoke are
  `[macOS-CPU advisory]`. Not a contest score, not a pointer row, not promotable.
* **GT lineage.** The vehicle pins the **PyAV** `gt_n600.npz`
  (`[[gt_n600_npz_is_pyav_lineage_train_on_dali_20260903]]`). δ_c is measured on that lineage because
  it is the target the loss sees; using DALI areas inside a PyAV-targeted loss would be a second
  lever. The DALI cross-check agrees within the known 1.011–1.017× n32 offset. Absolute d_seg values
  here are **not** DALI-authority numbers.
* **`F_c` is a compressed time-varying quantity.** The dual's effective weight spans 0.007→5.0 over
  the run; the median over the excursion window is the constant, its p10–p90 spread (1.92× Lane /
  1.37× Movable) is the honest dispersion, and it was measured on the CONTROL — the treatment's own
  dual will differ once the cap changes the field it responds to.
* **The equilibrium is nominal, not exact.** The balance law presumes both forces in the same
  boundary-normal units; on this vehicle they are not (measured cap/recall gradient ratio 0.0125).
  The formula sets the scale; the measurement adjudicates.
* **n = 1 seed, one cell.** A single-lever race on seed 20260902. It can move the design; it cannot
  close the family. Seeds 20260903 / 20260904 are the sign-repeat.
* **The smoke is a MECHANISM check, not a verdict.** One update is not d_seg. It says the seed is
  consumed, the bytes change, the term is one-sided, and how loud the term is — nothing about where
  step 5,000 lands.

## NEXT_IF_RESUMED

* **`SEALED-AWAITING-MAIN-METAL-CLAIM`** — owner MAIN; store
  `…/ddm_qbr1_born_fairform_burn_prep/ng2_area_cap/launch/`; fire trigger: the chain has released the
  Metal AND ng1's warm cell has been adjudicated (they are separate levers and must not compose
  before both are read, `[[m164]]`). Copy sealed → authorized, bind claims, fire the command above.
* **`CONDITIONAL-LAMBDA-SCALE-RACE`** — fire trigger: falsifier 2 FAILS (the area at step 2,000
  exceeds its start ratio). Then the FORM is exonerated and the SCALE is the next single lever, with
  the measured parity multiplier **×80.05** as its upper bracket and the derived λ as its lower one.
  This is the A/B the law has owed since 2026-07-08.
* **`CONDITIONAL-WARM-TIMES-CAP-CELL`** — fire trigger: BOTH ng1's warm cell and this cap cell win
  their own falsifiers. Only then is the composition worth a cell, and it must be measured, never
  assumed additive (`[[m164]]`: union ≠ sum of legs, 3.705×).
* **`FIXED-TAU-TELEMETRY-EVERYWHERE`** — the row costs one sigmoid and it is now proven byte-neutral
  by measurement. Every QBR1-lineage cell should carry it; the live chain's four remaining cells
  cannot (they run from the sealed tree that predates it), which is itself a reason to read their
  histories with sd1's decomposition rather than at face value.

## DEAD-ENDS

* **The soft-mass area estimator is CLOSED for thin classes on this vehicle** — measured 2.4336×
  biased on Lane against a 1.0407× truth. Recorded in the law's excluded domain.
* **`delta = 0.25` is CLOSED for this vehicle** — its equilibrium sits above every ratio the object
  reaches. Recorded in the law's excluded domain.
* **Reading the area through the annealed `tau` is CLOSED** — it would import sd1's −40.54%
  schedule leg into the cap's own units.
* **"The cap cell is a same-pins twin of its control" is CLOSED as a framing** — a new loss term
  moves the pinned trainer by construction. The reproducible statement is same-START /
  same-schedule / same-EMA / same-selection with a pin set that moves exactly at the lever's file.
* **Re-scaling λ to the measured gradient ratio before the burn is CLOSED for this cell** — it would
  make falsifier 2 unfalsifiable. It is the conditional NEXT lever, not this one.
