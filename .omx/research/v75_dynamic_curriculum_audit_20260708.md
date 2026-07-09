# v7.5 DYNAMIC-CURRICULUM / COSTATE / ANTI-HARDCODED-EPOCHS AUDIT — 2026-07-08

**Pointer contest-CPU 0.19110 UNMOVED.** Everything here is MEANS. Read-only, $0 audit.
STORES CONSULTED: live run `levelset_n600_crucible_v6_run1_20260708T095730Z` (run.log +
costate_shadow.jsonl + launch.sh, sacred READ-ONLY) · `experiments/train_levelset_witness_realized_through_R_mlx.py`
(`_evt_resolve_seg_form` / `_stage_converged` / `_evt_readiness_row` / EventBackstopGate wiring L5723-5771) ·
`src/tac/witness_autoconfig.py` (`crucible_v7` base L2270-2320 + `crucible_v7_wiring_gaps` L2008) ·
`src/tac/witness_control/event_wirings.py` · `.omx/research/costate_controller_design_20260705.md` ·
`SPEC_v75_optimal_single_trunk_20260708.md`.

---

## VERDICT (lead): the operator's stated premise is FALSIFIED by the live log — but the real state is a THREE-WAY split, and the residual hardcoding is narrower than feared.

The prompt's premise ("`plateau_ok=False` through ep300, yet tau_softplus started at the HARDCODED
ep300 → event machinery WIRED but INERT") is **factually wrong on the live run**. The authoritative
log row:

```
{"stage": "curriculum_transition_fired", "from": "ce", "to": "tau_softplus",
 "epoch": 257, "trigger": "loss_plateau", "nucleus_gated": true, "nucleus_ready": true}
```

**CE→tau_softplus FIRED at ep257 via `loss_plateau`, 43 epochs BEFORE the 300 cap.** The event
trigger is NOT inert; it fired for real, on a measured plateau, past the measured nucleus gate. The
`handoff_readiness.plateau_ok=False` rows are a **verdict-cadence + stage-reset sampling artifact**
(root cause §2), not evidence the trigger never fired.

### Current state per transition (the three-way split)

| transition | live run (crucible_v6) | v7.5 sealed (crucible_v7) | truly hardcoded? |
|---|---|---|---|
| **CE→tau_softplus** (seg FORM) | EVENT — `--curriculum-event-triggered` + nucleus guard. **Fired ep257/plateau** (cap 300 backstop) | inherited (in `EventTriggeredCurriculum` DSL lever; not in v7 DELETED/PROGRAM_OWNED) | **NO** — event-driven, cap is backstop |
| **τ-VALUE anneal** (softmax-temp octave ladder) | CLOCK (`--tau-anneal-shape cosine_hold`, anneal-epochs denominator) | **EVENT** — `--tau-advance-mode event` (TauAdvanceController, self-paced octave ladder) | v6 clock / **v7 NO** |
| **Muon entry** | **CAP 726** — no `--muon-start-event` in v6 launch.sh | **EVENT** — `--muon-start-event powerlaw_meat` (tau-descent weak-KAM exhaustion + S2 REV-B nucleation positive-control); 726 = fail-safe backstop | v6 YES / **v7 NO** |
| **lane-band engage** | CAP 350 | **EVENT** — `--lane-band-start-event lane_nucleus` (#315 per-class born+formed); 500 backstop | v6 YES / **v7 NO** |
| **seg-chroma engage** | CAP 300 | **EVENT** — `--seg-chroma-boundary-start-event annulus_plateau` (#333 detector promoted to trigger); 450 backstop | v6 YES / **v7 NO** |
| **l7_softplus** | 3000 = "never" (measured DEFECT stage, demoted) | 3000 = never | intentional-never (fine) |

**Bottom line:** the live crucible_v6 run is a HYBRID (tau FORM event; τ-value + Muon + lane + chroma
on caps). **crucible_v7 (= v7.5) is already wired FULL event-mode** for all five transitions — the
`--*-start-event` co-emission at `witness_autoconfig.py:2282-2284` + `--tau-advance-mode event`
:2293 was BUILT by the operator override 2026-07-08 (the three sensor→start wirings that were the
OWED gaps in the pre-registered v7 memo). `crucible_v7_wiring_gaps()` now reports all three **WIRED**.
So "make v7.5 dynamic" is ~80% DONE in the sealed config; the remaining work is calibration +
telemetry + the costate ACTUATION layer, not net-new wiring.

---

## Q2 — WHY the `plateau_ok` telemetry stayed False through ep300 (root cause: CALIBRATION+SAMPLING, not by-design-inert)

There are **TWO plateau computations on DIFFERENT histories/cadences**, and they legitimately disagree:

1. **The ACTUAL trigger** (`_evt_resolve_seg_form._fire` → `_stage_converged`, trainer L2087-2115):
   runs **every epoch** on the dense in-stage per-epoch `ep_loss` history (`state["losses"]`), with
   `min_stage_epochs=250 / plateau_rel_eps / plateau_windows=4`. This crossed threshold at **ep257**
   and fired.
2. **The `handoff_readiness.plateau_ok` telemetry** (`_emit_verdict_row`/`_evt_readiness_row`, L6160):
   runs **only at verdict cadence (every 25 ep)**, calling the SAME `_stage_converged` on the SAME
   `_evt_state["losses"]`. Sampled at ep250 → **not yet plateaued** (slope still > eps at the 250
   sample). Next sample ep275 → stage already fired at 257, so `stage_start` was RESET to 257 and
   `losses` cleared; the new tau stage has only ~18 in-stage losses `< min_stage_epochs(250)` ⇒
   `_stage_converged` returns False by the `n < min_stage_epochs` guard. Same at ep300.

So `plateau_ok=False` at ep250 (pre-fire), ep275, ep300 (post-fire, fresh stage) is **exactly what a
correctly-functioning system prints** — there is simply **no verdict-cadence sample at ep257** where
it would have read True. It is a 25-epoch-cadence blind spot straddling the fire, compounded by the
stage-reset. It is a **telemetry legibility defect**, not a controller defect. (The `ready` field =
`plateau_ok AND nucleus_all_ok` is pure observability, **never read back into training/parity/resume**
— confirmed L6188-6192, L5701-5704.)

**Secondary (genuine) calibration observation:** `--curriculum-min-stage-epochs 250` with cap 300
gives the plateau detector an event window of only **[250, 300] = 50 epochs** of freedom before the
backstop. The fire at 257 used 7 of those 50. The min-stage FLOOR, not the plateau, dominates the
timing — so CE→tau is "event-driven inside a tight collar." This is the real semi-hardcoding, and it
is a hand-tuned literal (`250`), not a derived quantity.

---

## Q3/Q4 — CONCRETE CHANGES to make v7.5 truly costate-driven + event-based (ranked; with the actuation-safety split)

### The actuation-safety split (the CONTAINMENT boundary — binding)
Per `costate_controller_design_20260705.md` §5 + CLAUDE.md: costate Phase A (observe→estimate→
recommend) is LANDED and runs as SHADOW (live `costate_shadow.jsonl` shows `actuation: NONE`,
`axis: [macOS advisory] NON-PROMOTABLE`). Phase B actuation is **DESIGN-ONLY, operator-GO gated**,
and its **actuation surface is DSL argv emission for the NEXT run, NEVER in-run process control**
(enforced by `test_no_actuation_capability`). The correct split, therefore:
- **In-run stage transitions (CE→tau, Muon, lane, chroma):** actuate AUTONOMOUSLY via the
  **deterministic event sensors** (already pure, seeded, resume-faithful, unit-tested). These are NOT
  "costate autonomy" — they are the trainer's own build-3 bounded controller. Safe to fire in-run.
- **Cross-run config changes (thresholds, λ bands, exploration budget, mode flips clock↔event):**
  costate RECOMMENDS via ranked DSL-argv diffs; operator GO actuates. Never in-run.

### Ranked changes (highest-EV first)

1. **[telemetry, $0, do first] Fix the `plateau_ok` legibility defect.** Emit a `handoff_readiness`
   row (or a dedicated `curriculum_transition_fired` echo already exists) carrying the trigger's
   OWN plateau value **at the fired epoch**, and stamp every readiness row with `in_stage_epochs` +
   `stage_start` so a reader never mistakes a post-fire stage-reset False for "never plateaued." Also
   log the dense-epoch plateau slope at verdict cadence (not just the boolean). This is pure
   observability (byte-identical) and directly kills the confusion that generated this audit. →
   trainer `_evt_readiness_row` / the fire row; register the new field, no DSL change.

2. **[calibration, low-risk] Derive `curriculum_min_stage_epochs` instead of hardcoding 250.** The
   50-epoch collar is the real residual hardcoding. Replace the literal with a DERIVED floor (e.g. a
   fraction of `anneal_epochs`, or "min epochs to accumulate `plateau_windows` verdict points at
   `eval_every`") on the value-provenance ladder, so the event window widens/derives rather than
   pinning the fire to a narrow band under a hand-tuned constant. Keep the cap as fail-safe backstop.
   → DSL: this is a `WitnessProgram` schedule-spine derivation; register the derived floor as a
   provenanced value (SRC_DERIVED), not a bare literal.

3. **[launch decision] Confirm v7.5 launches in crucible_v7 EVENT mode (already wired).** The Muon/
   lane/chroma events are BUILT in crucible_v7 (:2282-2284). The `self_paced_tau_advance` memo
   RECOMMENDS the FIRST unified-L_τ run in CLOCK mode to isolate the unify-τ variable, then flip to
   EVENT for run-2 (a one-token change). So the anti-hardcoded-epochs target for τ-value is a
   **staged flip**, not a missing build. Council/seal makes the final mode call per SPEC §8.

4. **[Muon] the powerlaw_meat sensor is the direct replacement for `--muon-start-epoch 726`.**
   `muon_meat_event` fires on tau-descent weak-KAM exhaustion (`horizon_epochs`, `meat_floor`,
   `min_points=8`) GATED on the S2 REV-B `nucleation_complete` positive-control. 726 becomes the
   fail-safe backstop. This is the single highest-value anti-hardcoding swap for the score-critical
   finishing stage — activate by ensuring crucible_v7 emits it (it does) rather than the v6 cap-only
   launch. **Verify the positive-control gate is satisfiable** (nucleation_complete must actually
   become True on the birth-arm, else Muon only ever fires on the 726 backstop → silently hardcoded).

5. **[costate ACTUATION, Phase B, operator-GO] Promote costate from advisory to the CROSS-RUN
   config recommender.** Wire the DECIDE ranker's ΔS-per-cost lever ordering into a `--dry-run`
   DSL-argv diff that the operator reviews between runs (the design's `config_diffs` field is already
   in the shadow schema, currently `[]`). This keeps in-run transitions on the deterministic sensors
   (safe, autonomous) while giving the costate a real, gated actuation surface for the knobs that
   change config class. Do NOT let costate drive in-run stage transitions — that is a config-class
   change and violates CONTAINMENT.

### The anti-hardcoded-epochs target, resolved
`--tau-softplus-start-epoch 300` → event `_stage_converged` plateau (FIRED 257) + backstop cap.
`--muon-start-epoch 726` → `--muon-start-event powerlaw_meat` (+ S2 REV-B gate) + backstop cap.
`--lane-band-start-epoch` / `--seg-chroma-boundary-start-epoch` → `lane_nucleus` / `annulus_plateau`
events + backstop caps. `--tau-anneal-shape cosine_hold` (clock) → `--tau-advance-mode event`
octave ladder. **The remaining bare literal worth killing is `curriculum_min_stage_epochs=250`**
(change #2) — the caps themselves are correct as fail-safe backstops, not as drivers.
NCDE hit-detector (#344) and terminal solve-don't-train (#341/#342) were **NOT found in the code
tree** (grep of `src/` + `experiments/`) — they are design-stage, not a live lever, and are NOT
part of the v7.5 launch surface today.

---

## What does NOT need changing (anti-churn)
- The event sensors + EventBackstopGate + resume registry are pure, seeded, unit-tested, resume-
  faithful, byte-identical-when-OFF. No rework.
- l7 = "never" is a correct, measured demotion of a defect stage.
- Costate SHADOW is correctly advisory (`actuation: NONE`); its containment is source-scan enforced.
- The live crucible_v6 run is the sacred birth-arm measurement — do not touch it; the dynamic-
  curriculum improvements land in crucible_v7 / the launcher, not the running dir.

**Pointer 0.19110 UNMOVED.** No exact row moved; this is a MEANS-layer apparatus audit. The END is a
byte-closed `upstream/evaluate.py` n600 row < 0.19110.
