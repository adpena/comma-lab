# DDM NG4 — the CONTINUOUS-OBJECTIVE cell: carry r10's terminal objective state

**Date:** 2026-09-04
**Arm:** `ddm_ng4_continuous_objective`
**Axis:** `[seal + bounded macOS-CPU mechanism smoke only; no Metal, no Modal, no contest eval]`
**Disposition:** **SEALED / RE-ROOTED / VALIDATED INSIDE THE SEALED TREE / SMOKE HELD BY THE
GOVERNOR — MAIN fires.**

## Result first

The cell is sealed and the lever is exactly two carried states. But the charter named FOUR
restarted objective states, and **two of them are not restarts at all.** I measured each one
against the quantity that actually acts, and the correction runs in the direction that SHRINKS
this cell — which is why it is in the source, asserted by tests, and stated before the burn.

| state the bug-class note named | is it discontinuous in the quantity that ACTS? | carried? |
|---|---|---|
| **expected-flip τ** | **YES** — 0.05 → 0.15, a **3.0×** wider soft band (gradient channel) | **YES** |
| **duals λ_Lane/λ_Movable** | **YES** — r10's converged pair → 0 (gradient channel) | **YES** |
| EMA law | **NO** — the EXECUTED effective decay moves 0.9991017964071857 → 0.9990793899844618, a relative gap of **2.24e-5**, and the EMA never touches the gradient | no |
| batch geometry | **NO** — `qbt.SELECTION_IDS == r10["pair_ids"]` is **True** and `chunk_pairs == 16` on both sides | no |

Three corrections I owe, each to something this arm was TOLD:

1. **"the EMA law changes (r10 0.99954 → burn 0.99908)" compares a TARGET to an EXECUTED value.**
   0.9995405077759483 is r10's *target* decay, which r10 never reached: its checkpoint carries
   `ema.warmup = True` and `ema.num_updates = 10010`, so the decay it EXECUTED at its last update
   was `min(0.99954, 10011/10020) = 0.9991017964071857` — the warmup ramp was still the binding
   term. The cell executes a constant 0.9990793899844618 from update 1. **The averaging rate is
   continuous to 2.24e-5.** And it could not have caused the excursion anyway: `_evaluate_milestone`
   runs inside `qbt.ema_scope` and re-encodes from `ema.shadow`, while `ema.update(model)` only
   ever writes the shadow — **the EMA is a MEASUREMENT channel, not a gradient channel.** Holding
   it identical to the control is what keeps that channel MATCHED, which is what makes the S_hat
   comparison valid at all.

2. **The batch geometry never restarted.** r10's authorized config carries the same 32 pair IDs
   `qbt.SELECTION_IDS` holds and the same `chunk_pairs = 16`. The only geometry item that differs
   is the seeded chunk ORDER, and it is deliberately held to the CONTROL's seed 20260902 so the
   pair isolates the objective.

3. **The cell moves τ_START only.** The burn's schedule already ENDED at 0.05 — r10's terminal
   temperature. It re-opened the band 3× wide to walk back to the value it started from. The seal
   receipt shows this directly: `expected_flip_tau_end` is **not** in `differing_keys`.

So the honest headline is narrower and sharper than the charter's: **the QBR1 stage entry restarts
TWO objective states, both on the gradient, and this cell continues both.** τ is held at r10's
terminal value; λ starts from r10's terminal multipliers. The optimizer stays COLD (`resume_from`
is null, exactly as the control), so the single lever is objective continuity and ng1's
warm-moment lever — which was RACED and LOST — is not composed with it (`[[m164]]`).

## The r10 terminal-state table (every value read from a sha-pinned artifact)

| quantity | value | source | label |
|---|---|---|---|
| `expected_flip_tau_start` | 0.15 | `AUTHORIZED_N32_R10_10020_20260829.json` (pin `r10_config`, sha `87eff6e8…`) | MEASURED |
| `expected_flip_tau_end` | 0.05 | same | MEASURED |
| `margin_steps` | 10,000 | same | MEASURED |
| **terminal τ** | **0.05** | RE-DERIVED: `qbt.tau_for_step(9999, 10000, 0.15, 0.05)` | DERIVED |
| — cross-check | float32 image `0.05000000074505806` | r10 `history_journal.jsonl` last `kind=row`, `objective.tau`, `stage_step 10000 / step 10010` | MEASURED |
| **λ_Lane** | **0.005040981907324784** | `stage_03_end.pt` (pin `r10_checkpoint`, sha `09fd4165…`) `curriculum_state.margin_constraint_state.lambdas` | MEASURED |
| **λ_Movable** | **0.017331143732962344** | same | MEASURED |
| — cross-check | identical to the journal's terminal `margin_constraint.lambda_after` | `history_journal.jsonl` | MEASURED |
| dual bounds | `{Lane 0.12, Movable 0.009}` | r10 config **and** `qbt.MARGIN_CONSTRAINT_MODE_PINS` — IDENTICAL | MEASURED |
| dual step size `eta_lambda` | 0.11387788414126129 | r10 config **and** the cell's pins — IDENTICAL | MEASURED |
| dual mode | `lane_movable_werr_primal_dual` | both sides | MEASURED |
| EMA target decay | 0.9995405077759483 (**never reached**) | `stage_03_end.pt` `ema.decay` | MEASURED |
| EMA warmup / updates | `True` / 10,010 | `stage_03_end.pt` `ema.warmup`, `ema.num_updates` | MEASURED |
| EMA **executed** decay | 0.9991017964071857 | `min(decay, (1+t)/(10+t))`, `tac.training.EMA.effective_decay` | DERIVED (matches r10's journalled `ema_effective_decay` to all 16 digits) |
| `chunk_pairs` | 16 | r10 config; `qbt.REAL_TRAIN_CHUNK_PAIRS` is 16 | MEASURED |
| `pair_ids` | the 32 n32 IDs | r10 config `== qbt.SELECTION_IDS` | MEASURED |
| learning rate | 2.0e-4 | r10 config; the cell's LR is the same, and there is no scheduler anywhere (ng1) | MEASURED |

**No decimal above is retyped into executable code.** `test_no_executable_source_retypes_r10s_terminal_decimals`
parses each touched file, strips docstrings, and fails if any of the four r10 decimals appears in a
literal. The τ the trainer reads is re-derived through the anneal geometry on every validate call;
the λ the trainer reads is re-read from the checkpoint on every validate call.

## Why the temperature restart is the suspect, in one line each

* `ddm_gm1` MEASURED at n600 that at τ = 0.15, **85%** of the seg gradient lands on already-correct
  pixels — 77.7% of it outside `m_safe`, where the uint8 round trip cannot flip anything.
* `ddm_md1` MEASURED that the cold-transition damage is **complete within 16 updates**, which is
  what a 3×-wider soft band does to a converged margin field in a few full-`lr` steps.
* `ddm_ng1` carried the optimizer moments ALONE and still restarted the objective; it damped the
  onset (−0.027 @1k) but ended **+0.0186 S_hat WORSE** than the cold control. The moments shape the
  transient; they are not its cause.

This cell is the missing CONTROL for all three: "just keep training r10, with nothing restarted but
the seed's batch order." If the excursion vanishes, the whole cold-transition family was an
objective restart.

## How the two states enter, and the gates they now pass

Three layers, the same permissive-DSL / strict-validator composition ng3 established.

1. **DSL (the lever).** `ContinuousObjectiveFromR10(r10_config_path, r10_checkpoint_path)` in
   `src/tac/witness_dsl/curriculum_dsl.py` reads both pinned artifacts, re-derives the terminal
   temperature through the anneal, extracts the terminal duals, and emits two namespaces
   (`expected_flip_tau.*`, `margin_dual.*`). `compile_qbr1_continuous_objective_config` turns them
   into the `tau_band` block, the two top-level tau scalars the trainer reads, the `margin_dual`
   provenance block, and `margin_constraints.initial_lambdas` — the executable multipliers.
   Both required arguments are PATHS: no machine-specific location is hardcoded in the DSL.
2. **The QBR1 gates (new).** `validate_tau_band_block` gained an `r10_continuation` branch and
   `validate_margin_dual_block` is new (`ddm_qbr1_born_fairform_burn_prep.py`, both called from
   `validate_config`). Each admits exactly two states: **no block ⇒ the historical form** (so every
   QBR1 cell sealed before ng4 — the live chain's, ng2's, ng3's, the cold control's — validates
   unchanged), or **a block ⇒ every value re-read from r10's pinned artifacts on this call**, with
   the trainer-read scalars required to equal the block's own. A hand-edited temperature or a
   hand-edited multiplier is refused, exactly as `validate_area_cap_block` refuses a hand-edited
   stiffness. The dual gate additionally refuses a multiplier carried across a **changed bound,
   step size or mode** — a multiplier whose constraint moved is a number whose meaning moved, which
   is the cross-regime constant transfer (`[[m143]]`) this cell exists to cure.
3. **The trainer gate (widened, not bypassed).** `admissible_expected_flip_tau_bands()` returns the
   legacy pair, ng3's law-resolved band, and the held band. The held band introduces **no new
   literal**: it is `(LEGACY[1], LEGACY[1])`, because r10 annealed to `LEGACY[1]`. That r10's
   terminal τ really equals `LEGACY[1]` is proved by the STRICT gate against r10's own config, not
   by the permissive one. A fourth band is still refused.

**One geometry contract moved and it is worth naming.** `qbt.tau_for_step` refused `start == end`
(`not start > end > 0`), so a held temperature was **structurally inexpressible** — the trainer
would have raised at update 0. It now admits `start >= end > 0`: a held temperature is the
degenerate linear anneal and the closed form is exact for it. This is not a loosening of the real
gate — the admissible SET is what enumerates bands, and an INCREASING band is still refused
(`test_tau_for_step_still_refuses_an_increasing_or_nonpositive_band`).

**The seeding change is one line on each of the two executable paths.**
`initial_margin_constraint_lambdas(config, bounds)` replaces `dict.fromkeys(bounds, 0.0)` in BOTH
`run_config` and `_run_resume_smoke_segment`. An absent `initial_lambdas` key reproduces the old
seeding exactly — that is what keeps a control cell byte-identical — and putting it on the smoke
path too is what stops this lever from being one the smoke cannot reach. A lever the smoke cannot
reach is an inert lever, which is the fake-implementation class this arm must not ship.

## The single lever, proven

Both configs compile fresh through `qbr1.compile_cell` in this tree, so their pins are this tree's
pins and only the lever separates them.

```
differing keys : ["cell_id", "expected_flip_tau_start", "margin_constraints",
                  "margin_dual", "output", "tau_band"]
allowed        : the six above plus "expected_flip_tau_end" (unused — the ends already agree)
margin_constraints fields moved : ["initial_lambdas"]   (bounds, eta_lambda, mode untouched)
executable dual seeding : cell {Lane 0.005040981907324784, Movable 0.017331143732962344}
                          control {Lane 0.0, Movable 0.0}
```

Held identical and asserted: `objective` (native_interface_weight 100), `ema`, `schedule`,
`initial_state`, `learning_rate` 2e-4, `pair_ids`, `selection_weights`, `total_steps` 5,000,
`milestones`, `seed` 20260902, `chunk_pairs` 16, `checkpoint_every_steps`, `device`, `source_pins`,
and `resume_from` = null (**COLD OPTIMIZER** — the same optimizer transition the control took).
`area_cap` is absent and the seal refuses a cell that carries one: ng2's cap, ng3's band and this
continuation are three separate levers, and union is not the sum of legs (`[[m164]]`, 3.705×).

### The pin delta

| pin | sealed control (`106d0dd0…`) | ng4 |
|---|---|---|
| `qbt_trainer` | `6eda9c20…` | `8de4112c…` |

ng2 moved this pin (a new loss term), ng3 moved it again (the widened tau check), MAIN re-pinned
`qbt_packet_schema`, and ng4 moves it once more (the geometry contract + the third band). A
same-pins twin was off the table before this arm existed. What licenses the comparison is ng3's
MEASUREMENT, not an argument: ng3's control step-1 trained state reproduced **ng1's pre-telemetry
cold reference sha bit-for-bit**, so every byte landed since ng1 is score-neutral on the training
path. **This arm re-measures that claim in its own smoke** (row below) — it is the only thing that
lets a cell on a moved pin be read against a control that ran under an older trainer.

## Seal receipt

| artifact | value |
|---|---|
| **sealed cell config** | `…/ddm_ng4_continuous_objective/sealed_configs/seed_20260902_continuous_objective_control_native100.json`, sha256 `8c17c2fe264752c7f87f2a40603139a365e902e7e8fad64d6b13f02dd4098215`, 11,815 B |
| **RE-ROOTED config — the one MAIN fires** | `…/sealed_configs/seed_20260902_continuous_objective_control_native100.rerooted.json`, sha256 `93f82b126dfddebc49e32ffd49734329c8d19142bbaa61450eff8f21e19b9267`, 12,263 B |
| sealed source tree | `/Volumes/VertigoDataTier/pact/ddm_ng4_continuous_objective/sealed_source_50e2cd2808/` at revision `50e2cd280810a2ec74c86ba26e923fa52fbcd217` |
| pins verify INSIDE the sealed tree | **PASS** (20 pins, its own interpreter, its own `REPO`) |
| pin re-root | 8 of 20 pin paths re-rooted; **every pin's sha256 + bytes identical** (the tool refuses otherwise) |
| **the sealed tree's own `validate_config` on the re-rooted config** | **PASS** — `tau [0.05, 0.05]`, `initial_lambdas {Lane 0.005040981907324784, Movable 0.017331143732962344}` |
| recompile determinism | every block this arm owns is byte-stable; only `ema.lawref.resolved_at` moves, and `qbt.stable_ema_law_identity` is the lineage's sanctioned comparator |

**Verify the config by hashing the FILE, never by recompiling** — a recompile legitimately moves
`ema.lawref.resolved_at`, which the QBR1 lineage has always kept inside the config.

The re-root + in-tree validation is the part that matters. A seal that has not been validated by
the interpreter that fires it is a claim, not a receipt
(`[[seal_validates_only_inside_the_tree_that_fires_it_20260904]]`); ng2's and ng3's configs were
refused by their own sealed trees for a pure PATH difference until MAIN built
`experiments/ddm_reseal_pins_inside_sealed_tree.py`. This arm ran it as part of `seal` and then
re-validated inside the tree, so the receipt above is the sealed tree speaking, not this arm.

## The cold control of record (read LIVE, never a retyped decimal)

`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100`.
Axis `[macOS-MPS n32 stratified advisory; not contest authority]`.

| step | S_hat | d_seg_hat | d_pose_hat | archive bytes |
|---:|---|---|---|---:|
| 0 | 0.39876797285867277 | 0.002518335978190104 | 0.0005757456120606528 | 106,714 |
| 1,000 | 0.46687521208987615 | 0.003051122029622396 | 0.0008233354187810106 | 106,667 |
| 2,000 | **0.48567677825279465** | 0.0032170613606770835 | 0.000864393511532432 | 106,626 |
| 3,000 | 0.47538291701253005 | 0.003139241536458333 | 0.0008181846911522883 | 106,637 |
| 4,000 | 0.44219037073377010 | 0.0029336293538411457 | 0.0006051119375803525 | 106,687 |
| 5,000 | 0.42514878445269977 | 0.002758916219075521 | 0.0006122744215585018 | 106,643 |

Endpoint excess **+0.026380811594027 (+6.6156%)**, of which ng1 recomputed **91.20% is d_seg**.

## Pre-registered falsifiers (fixed before the burn)

1. **PRIMARY — the excursion must not open.**
   `S_hat(1,000) ≤ 0.40376797285867277` (start + 0.005) **AND** `S_hat(5,000) < 0.39876797285867277`.
   *If it fails:* the objective restart was not the (whole) cause; the family framing is refuted at
   FORMULATION scope for the born object, and the discontinuities this cell did NOT carry — the
   cold optimizer, and the collapsed live/shadow gap below — become the next suspects.

2. **The 16-update damage must be absent** — RE-SPECIFIED, and the re-specification is a WEAKENING
   I am stating before the burn. The charter asks for `d_seg_hat` at step 16 within 1.2× of step 0.
   **That number does not exist in this cell's retained set:** `MILESTONES` is validated as exactly
   `(0, 1000, 2000, 3000, 4000, 5000)` and adding a step-16 milestone would change the sealed
   schedule and the retained payload set — a second lever. The strongest instrument available at
   zero extra cost is `realized_within_class_error`, the exact-argmax quantity the loop already
   computes on the realized post-R logits and journals EVERY update. So: **the realized
   within-class error at update 16 must be within 1.2× of its update-1 value, for BOTH Lane and
   Movable**, read from `<run>/history.jsonl`. It is a per-class error rather than the HT-weighted
   all-class d_seg, so it is a weaker test than the charter asked for.
   *If it fails:* the 16-update damage has a source other than τ and λ — report the first-update
   parameter displacement beside it, since the cold optimizer's `lr`-sized sign step is the
   remaining candidate.

3. **The dual trajectory must be continuous.** `margin_constraint_lambdas` at update 1 within one
   `eta_lambda` step (0.11387788414126129) of `{Lane 0.005040981907324784, Movable 0.017331143732962344}`,
   and neither class re-warming from 0. **Concretely, PREDICTED before the burn from the shared
   step-1 field** (§ next section): `{Lane 0.005149361310460459, Movable 0.017413431405350965}` for
   ng4 against `{Lane 0.00010837940313567591, Movable 8.228767238862249e-05}` for the control.
   *If it fails:* the carried multipliers did not reach the loop — an inert lever, which is the
   fake-implementation class this arm must not ship.

**Read the DECOMPOSITION at every milestone, never the composite.** The control's endpoint excess
is 91.20% d_seg. A cell that "fixed" S_hat by moving bytes or pose would be a different finding.

## What the two carried states DO at update 1 — MEASURED at $0, on a shared field

The bounded smoke was held by the governor (next section), so I measured the mechanism on a field
that already exists: **ng3's retained control-arm update-1 payload**, opened READ-ONLY. It is the
identical cold control config, seed, start state and first chunk, so its step-1 field is the field
BOTH arms see at update 1 — only the loss read off it differs. The differential code path is the
sealed one; this is a dry run of it, and it will be re-run on ng4's own payload in the real smoke.

**The DIFFERENTIAL passes, and cross-checks against another arm's independently measured number.**
At the control's τ = 0.15 with zero duals, all 15 objective components are **bit-identical**
between the two configs, `loss_total` **1.0765775442123413** on both sides — which is the exact
value ng3 measured for the same quantity. So the ng4 config changes nothing in the objective
FUNCTION; it acts only through the two states the caller supplies.

| quantity at update 1, on the shared field | control | ng4 | ratio |
|---|---|---|---|
| τ | 0.15 | **0.05** | 3.0× narrower |
| realized within-class error, Lane | 0.12095171598904345 | same field | — |
| realized within-class error, Movable | 0.009722595726195155 | same field | — |
| λ_Lane after update 1 | 0.00010837940313567591 | **0.005149361310460459** | **47.5×** |
| λ_Movable after update 1 | 8.228767238862249e-05 | **0.017413431405350965** | **211.6×** |
| margin-constraint penalty | 0.0023067535366863012 | **0.08141779154539108** | **35.3×** |
| `seg_expected_flip_realized` | 0.005018208175897598 | **0.0029315962456166744** | 0.584× |
| `loss_total` at its own operating point | 1.078884243965149 | 0.7406730651855469 | — |

**Two things in that table were not what I expected, and both are corrections.**

1. **Both constraints are BINDING at update 1 even though r10 ended with both SLACK.** r10's
   terminal within-class errors were Lane 0.11357 (bound 0.12, residual −0.0064) and Movable
   0.006661 (bound 0.009, residual −0.0023). One update after the transition they are Lane
   0.120952 and Movable 0.009723 — **both ABOVE their bounds**. So the transition itself pushes the
   two constrained classes past the constraint within a single update. This kills the guess that
   the carried multipliers are a decaying transient: with the constraint binding, they GROW, and
   the 47.5× / 211.6× head start persists rather than washing out.
2. **The dual leg is not small.** The carried multipliers make the cell pay a **35.3×** larger
   constraint penalty at update 1. τ and λ are therefore both live levers, not one lever plus a
   rounding term — which is a reason to read the two-state cell as a package and to keep
   decomposition (follow-on #3) conditional on this cell winning.

**A free by-product: this cell removes ddm_sd1's τ-schedule reporting artefact entirely.** With
`start == end`, `tau_for_step` returns the same temperature at every update, so the schedule leg of
the reported surrogate is identically **0%** by construction. ng3 cut that artefact 4.76× (−41.58%
→ −8.74%); ng4 removes it. MEASURED consequence, visible in the table: the cell's annealed
`seg_expected_flip_realized` and its fixed-τ ruler `seg_expected_flip_realized_tau_ref` are the
**same number** (0.0029315962456166744), because the held τ *is* `EXPECTED_FLIP_TAU_REFERENCE`.
For this cell — and only this cell — a falling reported loss cannot be a schedule artefact.

**Scope:** one frozen 16-pair chunk at update 1, on ng3's retained field. It confirms the mechanism
and pins falsifier 3's expected values; it says nothing about where step 5,000 lands.

## Bounded CPU smoke — HELD BY THE GOVERNOR at seal time

The smoke is the charter's: a NO-OP DETECTOR (step-1 state ≠ the control's) and the DIFFERENTIAL
(at the control's τ = 0.15 and zero duals, the two configs' objectives are bit-for-bit equal),
plus the sister halves that the objective is neither τ-blind nor dual-blind, plus ng3's
training-path-unmoved re-measurement.

**It had not run when this memo was written.** `system_memory_governor.live_admission_decision`
REFUSED every projection — 18, 21, 25 and 42 GiB — because ng2's and ng3's cells were both live on
the Metal and system-used sat at **112.1 GiB against the adaptive 116.0 GiB ceiling (headroom
3.9 GiB)**. That is the governor working, and a REFUSE is information, not an obstacle.

**A projection correction I made against my own first plan.** I split the smoke into one arm per
process (`smoke --arm {control,continuous}`, with `smoke-finalize` paying no forward pass at all —
it reads the payloads the arms already retained) on the assumption that ng3's **41.48 GiB**
high-water was two arms summed, so one arm would cost ~half. Then I read ng2's: **41.46 GiB**. Two
arms with *different* extra computations (ng2 a gradient probe, ng3 a differential) landing within
0.02 GiB of each other says the high-water is set by the ONE update's forward+backward, which is
identical in both — **not** by summing arms. So the split buys ISOLATION, not a smaller peak, and
arm 1 projects the full **42 GiB**. Over-projecting costs wall clock; under-projecting risks an OOM
cascade that would kill two live 3-hour Metal cells. Arm 2's projection is then derived from arm
1's MEASURED `peak_rss_bytes` + 15%, because by then a measurement exists.

A governed waiter (`…/ddm_ng4_continuous_objective/run_bounded_smoke_when_admitted.sh`, launched
through `tools/launch_detached_process.py` — a hand-rolled `nohup … & disown` was correctly refused
by `tools/launch_guard_hook.py` as the rc=144 kill class, and the first attempt died to exactly
that reaper) polls the same admission gate and fires each arm through the launcher the moment it
admits. It never overrides the governor.

**What is already MEASURED and what is still owed, stated separately:**

| smoke component | state |
|---|---|
| **DIFFERENTIAL** — all 15 objective components bit-identical at τ = 0.15 with zero duals | **MEASURED** (§ previous section), and the value `1.0765775442123413` cross-checks ng3's independently measured number for the same quantity |
| objective is neither τ-blind nor dual-blind | **MEASURED** |
| carried duals reach the loop | **MEASURED at the seeder** (`initial_margin_constraint_lambdas`, both executable paths, asserted by tests) — not yet through a real update |
| **NO-OP DETECTOR** — the two arms' step-1 trained-state sha256 differ | **OWED.** It needs one real B=16 update per arm and nothing smaller is the real path |
| ng3's training-path-unmoved re-measurement (control step-1 sha vs ng1's cold reference) | **OWED**, same update |

**DERIVED in the meantime, and labelled as such:** on the identical step-1 field the two arms'
losses at their own operating points are 1.078884243965149 and 0.7406730651855469. Both τ and λ
multiply θ-dependent terms (the seg surrogate and the per-class penalty), so the two gradients
cannot coincide and the step-1 states must differ. That is an argument, not a receipt — the sha
comparison is what settles it, and it is owed.

> **SMOKE RESULT — appended when the governed waiter completes; see
> `…/ng4_continuous_objective/bounded_smoke/BOUNDED_SMOKE_RESULT.json` and the waiter's receipt
> `NG4_SMOKE_WAITER2_DONE.json`. At seal time ng2 stood at 2,928/5,000 updates and ng3 at
> 887/5,000, both contending for the Metal, so admission at 42 GiB was ~35.5 GiB short. Whoever
> harvests this arm should read that JSON and append its rows here.**

## MAIN fire command

Preconditions MAIN owns: ng1, ng2 and ng3 are adjudicated (four separate levers must not compose
before each is read), the Metal is free enough for the memory-guard admission, a live scorer claim
and a live Metal claim exist, and the sealed tree is unchanged. MAIN copies the **re-rooted**
config to `authorized_configs/`, sets `launch_authorized: true`, and binds both claim IDs — all
three fields are in `config_identity`'s ignored set, so binding them does not disturb the sealed
identity.

```bash
shasum -a 256 /Volumes/APDataStore/pact/ddm_ng4_continuous_objective/sealed_configs/seed_20260902_continuous_objective_control_native100.rerooted.json
# expect 93f82b126dfddebc49e32ffd49734329c8d19142bbaa61450eff8f21e19b9267  (12,263 B)
```

```bash
SRC=/Volumes/VertigoDataTier/pact/ddm_ng4_continuous_objective/sealed_source_50e2cd2808
$SRC/.venv/bin/python $SRC/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng4_continuous_objective/launch/seed_20260902_continuous_objective_control_native100 \
  --cwd $SRC \
  --purpose "NG4 continuous-objective cell seed_20260902_continuous_objective_control_native100" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib 2.3959503173828125 \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt ng4_continuous_DONE.json \
  -- $SRC/.venv/bin/python $SRC/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config /Volumes/APDataStore/pact/ddm_ng4_continuous_objective/authorized_configs/seed_20260902_continuous_objective_control_native100.json
```

**Memory-guard admission:** `--measured-peak-rss-gib 2.3959503173828125` is the value ng2's and
ng3's live cells run under — the MEASURED per-cell Metal footprint of this identical object, not a
projection. `safe_run` re-runs `system_memory_governor.live_admission_decision` at spawn and
REFUSES (rc=5) if the system total would exceed the adaptive ceiling. At seal time it was
refusing, so MAIN should expect to fire after a live cell releases.

**Done-receipt name `ng4_continuous_DONE.json`** — distinct from every reserved name
(`ng2_area_cap_r2_DONE.json`, `ng3_tau_band_DONE.json`, `NG4_SMOKE_*_DONE.json`); the launcher
refuses a duplicate, which is how ng3's first smoke attempt was caught.

Cost: one cell, **~2.95 h** measured on the identical object, **~1.375 GB** retained.
**No control re-burn** — the control is the already-measured seed-20260902 row above.

**A storage fact MAIN should have, and it is trending the wrong way.** APDataStore held **16 GiB
free** when this arm sealed, against the 8 GiB reserve every cell's `storage_preflight` enforces.
ng2 recorded 23 GiB at its seal and ng3 recorded 22 GiB — so the tier is losing roughly 7 GiB per
generation. ng2's and ng3's cells will retain ~1.375 GB each, ng4's smoke ~0.3 GB and ng4's cell
another ~1.375 GB: about 3.2 GB against ~8 GiB of usable margin. It clears, but the NEXT generation
will not without a cold-store sweep. This arm's own footprint is 6 MB on APDataStore plus a 2.0 GB
sealed source tree on VertigoDataTier (166 GiB free), and that tree is exactly reproducible by
`git archive 50e2cd280810a2ec74c86ba26e923fa52fbcd217` — the revision IS the certificate, so it is
losslessly deletable once the cell has burned.

## Custody (ALWAYS KEEP THE PAYLOAD)

| artifact | path |
|---|---|
| seal receipt (r10 table, discontinuity audit, single-lever diff, determinism, falsifiers) | `/Volumes/APDataStore/pact/ddm_ng4_continuous_objective/SEAL_RECEIPT.json` |
| **sealed cell config** — sha `8c17c2fe…`, 11,815 B | `…/ddm_ng4_continuous_objective/sealed_configs/seed_20260902_continuous_objective_control_native100.json` |
| **RE-ROOTED config (fire THIS)** — sha `93f82b12…`, 12,263 B | `…/sealed_configs/…rerooted.json` |
| pin re-root receipt | `…/ddm_ng4_continuous_objective/PIN_REROOT_RECEIPT.json` |
| matched control (reference recompile, marked `DO_NOT_FIRE`) | `…/sealed_configs/matched_control_of_record_seed_20260902_control_native100.reference.json` |
| sealed source manifest | `…/ddm_ng4_continuous_objective/SEALED_SOURCE_MANIFEST.json` |
| sealed source tree (rev `50e2cd2808…`, pins verify inside) | `/Volumes/VertigoDataTier/pact/ddm_ng4_continuous_objective/sealed_source_50e2cd2808/` |
| governed smoke waiter + log | `…/ddm_ng4_continuous_objective/run_bounded_smoke_when_admitted.sh`, `…/smoke_waiter.log` |
| bounded smoke (per-arm receipts + retained payloads) | `…/ddm_qbr1_born_fairform_burn_prep/ng4_continuous_objective/bounded_smoke/` |
| run output root (empty; MAIN's cell writes here) | `…/ng4_continuous_objective/runs/seed_20260902_continuous_objective_control_native100` |
| code + 32 tests | `experiments/ddm_ng4_continuous_objective_cell.py`, `src/tac/tests/test_ddm_ng4_continuous_objective_cell.py` |

`authorized_configs/` is **not** written by this arm, and a test asserts the arm's source contains
no path that could write it, no reference to the claims ledger, and no `"mps"` literal. Nothing was
written under ng2's or ng3's run roots, `authorized_configs/` or `CHAIN_LEDGER.jsonl`; the claims
ledger was not touched; the seed-20260902 control run was opened read-only. 0 Metal / 0 Modal /
0 contest-eval invocations, $0.

## Equations leg (`tac.canonical_equations`)

**`muon_finisher_schedule_warmstart_and_lr_anneal_v1` — CONSUMED as the SIBLING, not the law.**
That law is the OPTIMIZER-side statement of this transition: its anchors record a cold-start
d_seg spike of +0.000357 and a +27.5% quench that never re-beat the pre-transition best, and its
cure is warm-start momentum plus an LR anneal. **This cell is its objective-side sibling** — same
transition, same shape of damage, but the carried state is the objective's (τ, λ) rather than the
optimizer's, and there is no LR schedule in this trainer to anneal (ng1 verified at source). **No
anchor appended, and the law is NOT extended to cover this cell:** its vehicle is the Muon
finisher, ours is QBF1-born on a frozen SegNet, and ng1 already MEASURED that this law's own cure
(warm moments) LOST on our vehicle by +0.0186 S_hat. Citing it as a prediction for ng4 would be the
ancestor-number transfer this campaign keeps extincting (`[[L18]]`, `[[m143]]`).

**`checkpoint_trajectory_error_partition_v1` — CONSUMED IN-DOMAIN as the reading rule.** Both of
its anchors are md1's partitions of THIS object (the QBR1 cold control and ng1's warm cell, seed
20260902, 71 checkpoints, 6,291,456 sites, DALI GT authority, persistent fraction 0.9). Its
`domain_of_validity.included` names "reading the persistent share as a CEILING on
optimizer/schedule credit", and that is exactly how falsifier 2 must be read: a schedule/objective
lever can only reach the CHURN mass, so a continuation cell that removes the excursion is capped
by that partition and must never be reported as if it could move the persistent share. **No anchor
appended** — this cell has not run, and its `excluded` clause explicitly refuses "transferring a
measured persistent SHARE across cadences". When ng4 burns, md1's sweep at the SAME cadence is the
instrument that earns an anchor.

**`margin_band_satisficing_threshold_v1` — NOT consumed, deliberately.** ng3's band is the
law-resolved `(m_safe, δ_R)`; ng4's held τ is r10's own terminal value and cites no law. Dressing a
MEASURED anchor in a law's name would be false provenance. The held band's provenance rung is
"measured anchor with content-hashed artifacts" (Catalog #351 class 2), and the validator re-reads
those artifacts rather than trusting the block.

**`ema_decay_run_geometry_v1`** is consumed unchanged and IN-DOMAIN: the cell inherits the
control's sealed decay 0.9990793899844618 and the strict
`check_ema_executable_law_matches_sealed_law` gate sees no change. Its `lawref.resolved_at` is the
volatile field discussed above.

**FORMALIZATION_PENDING** — the law this cell's headline would need does not exist:

> *At a training-stage boundary on a converged piecewise-constant argmax field, the transition is
> discontinuous in every ANNEALED or ADAPTED objective state that is not explicitly carried, and
> the damage is proportional to the discontinuity in the states that enter the GRADIENT, not to the
> number of states restarted. States that enter only the MEASUREMENT channel (a weight EMA read at
> milestones) cannot cause the excursion and must be held MATCHED to the control rather than
> carried.*

It should be registered once a continuation cell has burned, so it anchors on the cure rather than
on this design.

## Scope and limits (these travel with the numbers)

* **Axis.** Every `S_hat` quoted is `[macOS-MPS n32 stratified advisory]` (the control's own
  retained milestones). The seal and smoke are `[macOS-CPU advisory]`. **No score claim, nothing
  promotable, the pointer is untouched.**
* **GT lineage.** The vehicle pins the **PyAV** `gt_n600.npz`
  (`[[gt_n600_npz_is_pyav_lineage_train_on_dali_20260903]]`). Both arms sit on the identical
  lineage, so the comparison is internally valid; the ABSOLUTE d_seg values are not DALI-authority
  numbers. md1's partition (cited above) used the DALI cache, so its shares and this cell's
  milestones are on different GT lineages and must not be arithmetically combined.
* **The live/shadow gap is a residual discontinuity this cell does NOT close.** r10's live weights
  and its EMA shadow differ by 8.4488e-03 relative (ng1); the cell starts BOTH at the shadow,
  collapsing the gap. Closing it would move the START POINT — a different lever — and would break
  comparability with the control, because `build_initial_state` is the pinned same-start for every
  cell in this generation. Named, queued, not closed.
* **n = 1 seed, one cell.** A single-lever race on seed 20260902. It can move the design; it cannot
  close the family. Seeds 20260903 / 20260904 are the sign-repeat.
* **The re-specified falsifier 2 is weaker than the charter's**, and it is stated above rather than
  quietly substituted.
* **`EXPECTED_FLIP_TAU_REFERENCE` stays at 0.05**, which is now exactly the cell's own held τ. The
  fixed-τ telemetry row (ng2) therefore reads the SAME value as the annealed row in this cell and
  differs in the control — that is a property of the cell, not a bug, and the smoke measures both.

## NEXT_IF_RESUMED — every row carries a disposition, an owner and a fire condition

| # | follow-on | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **`SEALED-AWAITING-SMOKE-THEN-MAIN`** — finish the bounded smoke (the waiter is armed), then copy re-rooted → authorized, bind claims, fire the command above | **SEALED, ready** | this arm for the smoke; MAIN for the burn | the governor admits (a live cell releases) AND ng1/ng2/ng3 are adjudicated |
| 2 | **`CONDITIONAL-WARM-AND-CONTINUOUS-TWIN`** — carry r10's AdamW moments AND its objective state | **QUEUED-WITH-FIRE-ORDER** | MAIN to assign | fires ONLY if falsifier 1 PASSES; composing a lost lever (ng1) with a winning one is only legible once the winner is measured alone (`[[m164]]`) |
| 3 | **`CONDITIONAL-DECOMPOSE-THE-TWO-CARRIED-STATES`** — τ-only and λ-only cells | **QUEUED, no fire order** | unowned; MAIN to assign or close | fires if falsifier 1 PASSES and the effect is large enough that attributing it to one of the two is worth two more cells; if it FAILS, decomposition is wasted |
| 4 | **`CLOSE-THE-LIVE-SHADOW-GAP`** — start from r10's LIVE weights with its shadow as the EMA state, instead of collapsing both to the shadow | **QUEUED, needs a control** | unowned | fires only with a NEW matched control, because it moves the start point and every existing cell's comparability rests on the pinned same-start |
| 5 | **`STAGE-ENTRY CENSUS`** — ng3 found a QBR1 config key with no validator; this arm found two more executable states (τ seeding, λ seeding) that no gate read. Every OTHER state carried across a stage boundary is the same shape of hole | **QUEUED, needs a census first** | unowned — naming it without owning it would be the deferral scatter this repo extincted (`[[m36]]`) | fires if a third stage-entry state is proposed for carrying |
| 6 | **`REGISTER-THE-STAGE-ENTRY-CONTINUITY-LAW`** — the FORMALIZATION_PENDING statement above | **QUEUED** | whoever harvests the continuation cell | fires when the cell returns, so it anchors on the cure |

## DEAD-ENDS

* **"the EMA law restarts 0.99954 → 0.99908" is CLOSED as a framing** — it compares r10's TARGET
  decay to the cell's EXECUTED constant. The executed-to-executed gap is 2.24e-5, and the EMA is a
  measurement channel that cannot cause the excursion.
* **"the batch geometry restarts" is CLOSED** — MEASURED identical on both sides.
* **"τ restarts on both ends" is CLOSED** — only the START moves; the burn already ENDED at r10's
  terminal temperature.
* **"a held temperature is expressible in the sealed config path" is CLOSED as a premise** —
  `tau_for_step` refused `start == end` and would have raised at update 0. The charter's fallback
  ("land it as a DSL-held lever… snapshot… re-root… validate inside") is the path this arm took.
* **Composing ng1's warm moments, ng2's area cap or ng3's band into this cell is CLOSED** — one
  lever, and the seal refuses a cell that carries an area cap.

## Two failing tests that are NOT this landing's (checked, not assumed)

The repo-wide DSL/lever sweep after this landing showed 9 failures. Seven reproduce at HEAD in a
detached `git worktree`. The remaining two were checked individually against HEAD with the live
`.omx` state symlinked in:

* `test_live_lever_activation::…[--film-row-dropout]` — **fails at HEAD too.** It is driven by the
  activation ledger's contents, and this landing adds no trainer flag.
* `test_spec_c1_optimal_form_20260715::test_expected_levers_and_manifest_records` — asserts
  `launch_blockers` is non-empty. `_derive_launch_blockers` reads the FILESYSTEM
  (`experiments/results/*/dry_start_report.json` and a `_sR.npz` sidecar), never the DSL. In this
  tree both slots are satisfied by pre-existing artifacts — a GREEN `c1_optimal_form` dry-start
  report from 2026-07-16 and `experiments/results/mlx_fleet_gt_cache/gt_n600_sR.npz` — so the empty
  list is correct and predates this arm. The worktree passed only because its
  `experiments/results` was nearly empty.

Both are recorded here rather than left as ambient red, because a warning nobody owns is the second
one that gets ignored.
