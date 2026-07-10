# v8 INCREMENT-1a KILL-GATE — RUN 2026-07-10 (the pre-#385 $0 decoupling screen)

**VERDICT (one line):** `REFUSED-arm-missing-or-toy` — the v8 inc-1a decoupling A/B is **NOT $0-rulable**: the matched-compute shared-head CONTROL arm is a **TRAINED artifact** (governed EVENT + owed-9 lateral-carrier BUILD) and was **NOT fabricated** (NO-FAKE). What DID land at $0: the analytic decoupled-arm floor **d_seg = 0.100403 (n600)** (reproduces the SPEC precedent EXACTLY), harness end-to-end validation, and a 3-replicate determinism confirmation (spread = 0 → operative floor = R7 3.46e-6). **Pointer 0.19110 UNMOVED — MEANS.**

---

## What ran, and at what level

Per `SPEC_v8.1_20260709.md` §11 (owed-9 BUILD) + the operator contingency clause, the SPEC'd
trained arms and the lateral-capable 3-curve carrier (`road_undriv_bulk_field` `x_L(y)`/`x_R(y)`)
**do not exist yet**, so the gate ran at the **ANALYTIC-GENERATOR level** it was sealed for,
labeled `[analytic-generators]`. `$0`, no GPU, no training, run dirs read-only.

| arm | ran? | level | mask d_seg (n600) |
|---|---|---|---|
| **decoupled_analytic** (per-class independent analytic fields: horizon poly + lane band + hood; ∂φ_c/∂θ_c′=0 by construction) | ✅ | `[analytic-generators, no-trained-fields]` | **0.100403** |
| **control_shared_head** (matched-compute TRAINED shared head) | ❌ NOT RUN | — | — (does not exist; not fabricated) |

- **config:** `derive_crucible_v8_inc1a_config(gt_n600, 600)` → `validate()==[]`, `unknown_count()==0`, 18 provenanced fields, `bc_mode=no_offset`, `no_offset_d_seg=0.0031436` (LawRef), `seed_replicates_per_arm=3`.
- **evaluator:** `tac.inc1a_harness.decoupling_screen.evaluate_kill` (the harness OWNS the verdict — REUSE, not re-derived). With `control=None` it REFUSES ("an arm is missing") — the honest NO-FAKE path.
- **per-class mask d_seg (decoupled analytic):** Road 0.0212 · Lane 0.3617 · Undrivable 0.1616 · Movable 1.0 (unmodeled → folds to Road, expected) · MyCar 0.0039. Undrivable carries 79.7% of flip mass (the F4 single-valued-horizon lateral under-coverage — exactly what owed-9's lateral 3-curve carrier is designed to home).
- **determinism / seed floor:** 3 deterministic replicates → d_seg identical → `seed_spread = 0.0`. This is a DETERMINISM confirmation, NOT a measured training-seed variance (that instrument is N/A at the analytic level — no training seed). Operative `delta_mask = max(3.46e-6 R7 floor, 0.0) = 3.46e-6`.

## What this CARVES (P10) for the #385 which-to-run decision

- The v8 kill-gate is **NOT a $0 pre-training screen** as the #385 framing hoped. The matched-compute
  control arm requires training, so **no CONFIRMED/KILLED ruling is reachable at $0**. The A/B is a
  necessary-condition partition screen that structurally needs the governed training EVENT.
- **Banked at $0** (real, n600, advisory/NON-PROMOTABLE): (a) the analytic decoupled-composite FLOOR
  `0.100403` — the number any trained Stage-A decoupled arm must beat; (b) harness end-to-end
  validation; (c) determinism.
- **Cost asymmetry for #385:** choosing **v8** commits to owed-9 (lateral 3-curve carrier BUILD) **+**
  a governed decoupled/control training EVENT **before any A/B ruling exists**. **v7.5.2** (a training
  launch) has no such pre-ruling BUILD gate — it is directly launchable. This is a material input to
  the which-to-run GO, surfaced honestly rather than papered over.

## Falsification scope

`verdict_scope: INSTANCE` — this REFUSED is a statement about THIS $0 analytic attempt, **not** a
FORMULATION kill of decoupling and **not** a paradigm verdict. Decoupling remains untested (the A/B
needs trained arms). `cannot_falsify` through-R survival (mask-optimal ≠ score-optimal; the SUFFICIENT
test is 1b). `flat_paint_confound = EXCLUDED_BY_CONSTRUCTION` (both arms paint-free).

## Artifacts (durable)

- result JSON: `experiments/results/inc1a_killgate_20260710/inc1a_killgate_result.json`
- verdict JSON (`tac.verdicts.emit_verdict`): `experiments/results/inc1a_killgate_20260710/inc1a_killgate_verdict.json`
- per-replicate checkpoints: `experiments/results/inc1a_killgate_20260710/replicate_{0,1,2}.json`
- runner: `experiments/results/inc1a_killgate_20260710/run_inc1a_killgate.py`
- bulletin: `gate_ruled` event, subject `#385` (`.omx/state/session_events.jsonl`)

## Triality legs

- **DAG:** FEED-1a-gate (`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`).
- **equations:** EmpiricalAnchor `v8_inc1a_killgate_ruling_analytic_20260710` appended to
  `v8_geometric_rate_decomposition_v1` (registered, 6 anchors).
- **DSL:** N/A (a measurement — no trainer lever / launch / curriculum change).

STORES CONSULTED: `SPEC_v8.1_20260709.md` · `derive_crucible_v8_inc1a_config` · `decoupling_screen.evaluate_kill` · `analytic_smoke` · `scaffold_assembler` · `mask_dseg_meter`. `$0` · no GPU · #205 STOPPED. Pointer 0.19110 UNMOVED — only a byte-closed `upstream/evaluate.py` n600 row < 0.19110 moves it.
