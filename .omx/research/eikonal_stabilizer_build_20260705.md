---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "the arbitration arms are n24 at accum 8 — the step COUNT matches the v4 horizon but the landscape is the 24-pair slice of the 600-pair basin; a STABLE arm here licenses the GO config only because the DISEASE reproduces on the same slice (control 24.2x). And the lambda_pre number is measured at the RESTORED state whose self-orient feats are the ep100-EMA vintage, not the vintage v4 trained on — the 38/eta falsification is a falsification AT THAT STATE, not of the law everywhere."
council_assumption_adversary_verdict:
  - assumption: "the eikonal runaway is an Adam-EoS crossing governed by lambda_pre* ~= 38/eta"
    classification: CARGO-CULTED
    rationale: "MEASURED inverted: lambda_pre = 3.66e6 at the ep100 restored state => pi_EoS = eta*lambda_pre/38 ~= 94 >> 1 even though lr 9.1e-5 is measured-stable (pi ~ 9.6). The quantitative law fails at this state; the eps-floor-dominated preconditioner coordinates + minibatch-vs-fullbatch sharpness + feats-vintage basin shift are the three named suspects. The DIRECTIONAL content (higher eta => runaway) survives."
  - assumption: "damping terms that stabilize |grad m| necessarily degrade boundary sharpness (d_seg)"
    classification: CARGO-CULTED
    rationale: "the arbitration reads d_seg at ep100/120/140 per arm to MEASURE it rather than assume it; StEik's design point is exactly normal-direction-only damping so tangential (dash/boundary) geometry is free."
council_decisions_recorded:
  - "op-routable #1 (GO-gated): relaunch config = the arbitration winner's extra-flag delta on the v4 argv (see §6)"
  - "op-routable #2: do NOT register eos_adam_preconditioned_threshold_v1 (anchor contradicts the in-window prediction; FORMALIZATION_PENDING with the negative recorded)"
related_deliberation_ids: [stepping_instability_diagnostic_20260705, litsweep_training_dynamics_control_20260705]
---

# EIKONAL-STABILIZER BUILD — StEik/ViscoReg cures + rollback guard + λ_pre probe (n24 arbitration)

**Axis discipline: every number below is `[n24 advisory — mechanism probe, NOT n600 evidence]`;
verdict d_seg rows are `[macOS-CPU advisory] NON-PROMOTABLE`. NOTHING launches at n600 — the
recommended config is operator-GO-gated. Pointer contest-CPU 0.19110 UNMOVED (this whole memo is
means/apparatus).**

Design source: `litsweep_training_dynamics_control_20260705.md` (StEik arXiv 2305.18414 ·
ViscoReg arXiv 2507.00412 · Adam-EoS λ_pre* ≈ 38/η · guard-blocks-self-stabilization) +
`stepping_instability_diagnostic_20260705.md` (the 5-arm mechanism matrix; snapshot
`bd_calib_20260705/snap/resume_state_ep100.npz`). Build commit: `89c2add13`.

## 1. What was built (all default-OFF; byte-identity PROVEN, not asserted)

| surface | flag(s) | mechanism |
|---|---|---|
| **StEik damping** (build 1a) | `--eikonal-steik-weight W` | ADDITIVE `W · mean\|∇m^T H(m) ∇m\|` (the paper's L1 directional-divergence, RAW gradient) on the decision margin m = φ_top1−φ_top2 — damps ONLY the normal-direction second-order mode (the proven anti-diffusive instability), tangential curvature (lane dashes) free |
| **ViscoReg viscosity** (build 1b) | `--eikonal-viscosity EPS` + `--eikonal-viscosity-anneal N` | REPLACES the eikonal residual with the viscous form `(|∇m|−1−ε·Δm)²` (p=2) while ε>0; linear ε→0 over N absolute epochs (the paper's vanishing-viscosity continuation — our Γ/GNC philosophy applied to the eikonal term itself) |
| **Rollback guard** (build 2) | `--spike-guard-mode rollback` (+ `--spike-rollback-{window,frac,lr-cut,max}`) | tolerate single finite spikes (STEP them — EoS oscillation IS the self-stabilization feedback the legacy guard was blocking); on SUSTAINED runaway (>frac of a FULL window) restore last-good weights+EMA+opt (moments RESTORED — measured 6.7× vs 25.3×), cut lr ×0.5 persistently, re-arm a fresh median. Budgeted (`--spike-rollback-max`, then legacy semantics, loud) |
| **λ_pre probe** (build 4) | `--lambda-pre-probe-iters N` (+ runner `experiments/probe_lambda_pre_hvp.py`) | Adam-preconditioned power iteration (forward-diff HVP over the full P-pair grad, fp64 accumulation, central-diff consistency row); preconditioner matches mlx AdamW exactly (√v+1e-8, bias-correction-aware); exits BEFORE any training step |
| **Arbitration arms** (build 3) | `probe_resume_stepping_instability.py` arms `steik_005/steik_05/steik_5/visco_03/visco_1/rollback_guard/steik05_rollback` | all at the UNSTABLE lr 1e-3 so a STABLE verdict is attributable to the cure; `--eval-every 20` reads d_seg drift |

**Byte-identity at OFF (the established A/B pattern, EXECUTED):** same argv (n1 CPU, 3 epochs,
bd 0.2 + persistence + amplify active) run on pre-change HEAD vs post-change tree →
**106/106 `levelset_resume_state.npz` keys byte-equal + deploy EMA npz byte-equal + per-batch
loss values identical** (rows differ only by the new always-present `eik_steik: 0.0` schema key).
Artifacts: `experiments/results/eik_stab_build_20260705/idref_{pre,post}` + `idref_argv.sh`.
Tests: `src/tac/tests/test_eikonal_stabilizer.py` (27: term math on analytic fields — linear
plane → 0, quadratic → closed form, random-field vs independent numpy oracle; schedule endpoints;
guard state machine incl. the induced-runaway scenario; flag/plumbing/fail-closed guards).
Micro-batch: stabilizers FAIL CLOSED under `--micro-batch-pairs>1` (silent-drop NO-FAKE class).

### Adaptation honesty (per NO-FAKE; the paper terms were lifted, then adapted — documented)
1. **Field**: both papers regularize a single SDF head u; ours is the decision margin m — the
   SAME field our eikonal drives to |∇m|=1 and the field the measured runaway lives on.
2. **Stencils**: discrete central differences on the (H−2,W−2) interior grid replace autograd
   Hessians (our margin is a grid field through R, not a cheap double-differentiable MLP output).
   The legacy eikonal keeps its forward-diff (H−1,W−1) stencil; at ε→0 exactly the code switches
   back to it (an O(stencil) discontinuity at the switch, documented, ε~0 by then).
3. **Schedules**: StEik's paper anneals its weight linearly to zero mid-training — ours is a
   constant flag (short probe horizon; schedule addable if the arm wins). ViscoReg's
   piecewise-linear ε-decay is implemented as a single linear decay (the paper reports
   insensitivity to the profile).
4. **ViscoReg REPLACES the residual** (same constraint, viscous form); composing it with
   `--eikonal-junction-relax` would silently drop the junction weight → fail-closed validator.

## 2. λ_pre HVP probe — the 38/η law test (HONEST NEGATIVE at this state)

Measured on the ep100 snapshot (n24 slice, moments RESTORED — v_norm 6.4e-3, opt_step 6837,
bias_correction False), 12 preconditioned power iterations, GPU, `[n24 advisory]`:

```
lambda_pre        = 3.663e6    (iter trace converged: 3.02e6 → 3.66e6 plateau by iter 9)
central-diff check= 3.230e6    (fwd-vs-central rel 0.134 — order-of-magnitude solid)
eta(ep101)        = 9.779e-4   (the as-scheduled resume lr)
pi_EoS            = eta·lambda_pre/38 = 94.3
prediction window = [4.2e4, 7.6e4]  →  NOT IN WINDOW (≈ 50–90× above)
law-implied eta_max = 38/lambda_pre ≈ 1.04e-5  — but lr 9.1e-5 is MEASURED-stable (pi ≈ 9.6)
```

**Verdict: the quantitative Adam-EoS bracket test FAILS at this state** — λ_pre is ~2 orders
above the 38/η prediction, and even the measured-stable lr violates the naive threshold.
`eos_adam_preconditioned_threshold_v1` stays **FORMALIZATION_PENDING, NOT registered** (the
anchor contradicts the in-window prediction; register-only-if-clean discipline). Three named
suspects for the gap, each testable later: (a) **ε-floor-dominated preconditioner coordinates** —
params with v≈0 get d=√(1e-8): a static-P power iteration amplifies exactly the coordinates
where the law's quasi-stationary-v assumption breaks (the 2506.04805 decoupling mechanism);
(b) **minibatch-vs-fullbatch sharpness** — the actual steps are 8-pair; the probe's H is the
24-pair full-batch loss; (c) **feats-vintage basin shift** — the restored state rebuilds
self-orient feats from the ep100 EMA argmax (5.6% flip vs the training vintage), i.e. the probed
basin is a sharper wall than the one v4 walked (restored-state loss 94 vs continuous ~7, per the
diagnostic). The DIRECTIONAL content (larger η ⇒ runaway; sharpening ⇒ threshold falls) is
unchallenged — what failed is the specific constant 38.

## 3. Arbitration control (the disease reproduces under the new code, flags OFF)

`baseline_v3` re-run at 40 epochs past resume (=120 optimizer steps at accum 8 — PAST the
~115-step horizon that killed v4), guard disarmed: **EXPLODES 24.2×** (trough 19.9@ep111 →
peak 482; runaway terms pose + eikonal; eikonal reaching ~415 raw-contribution at ep125+).
d_seg verdicts: 0.0710@ep100 (resume, phantom-β d_pose row known) → 0.0640@ep120. Report:
`experiments/results/eik_stab_arbitration_20260705/report_control_baseline_v3.json`.

## 4. ARBITRATION TABLE (n24, 40 ep past resume = 120 steps, all cure arms at the UNSTABLE lr 1e-3)

| arm | flags delta | verdict | trough → peak (ratio) | d_seg ep100→120→140 | reading |
|---|---|---|---|---|---|
| `baseline_v3` (control) | none | **EXPLODES** | 19.9@ep111 → 482 (**24.2×**) | 0.0710 → 0.0640 → (cut) | the disease, reproduced under the new code |
| `steik_005` | steik W=0.05 | **EXPLODES** | 68.7@ep114 → 39,536 (**575×**) | — | eik_steik ITSELF a runaway term |
| `steik_05` | steik W=0.5 | **EXPLODES** | 268@ep114 → 383,010 (**1431×**) | — | worse, monotone in W |
| `steik_5` | steik W=5.0 | **EXPLODES** | 3,751@ep112 → 1,691,483 (**451×**) | — | worst |
| **`visco_03`** | **visco ε=0.3** | **STABLE** | **15.4@ep130 → 18.4 (1.19×)** | **0.0710 → 0.0572 → 0.0308** | **THE WINNER: descends 75.7→18.4 over 120 steps at the killer lr; d_seg 2.3× BETTER; no dash-erasure signal (d_seg includes lane pixels and IMPROVES)** |
| `visco_1` | visco ε=1.0 | EXPLODES | 25.6@ep110 → 2,105 (82×) | — | over-viscous; ε has an upper stability edge too |
| `rollback_guard` | rollback mode (window 6, frac 0.5) | EXPLODES (deadlock GONE) | 19.1@ep111 → 486 (25.5×) | 0.0710 → 0.0693 → 0.0423 | 0 skips (absorbing state eliminated ✓); ONE rollback @ep129 (lr→4.8e-4) but it restored a POISONED ep128 snapshot — see reading below |
| `steik05_rollback` | steik 0.5 + rollback | EXPLODES | 358@ep113 → 188,507 (527×) | 0.0710 → 0.1171 → 0.0599 | steik dominates; combination not helpful |

**Reading (measured, per-arm):**
1. **ViscoReg ε=0.3 is the arbitration winner** — the ONLY stable arm at the lr that kills every
   other config, it DESCENDS (75.7 → trough 15.4 → 18.4; mild late rise 1.19×, under the 3×
   explode bar — keep the n600 creep gates armed), and its d_seg is the best measured
   (0.0308@ep140 vs control 0.0640@ep120). The smoothing-vs-sharpness risk did NOT materialize
   at this horizon: d_seg (lane pixels included) improves 2.3×. Theory-consistent: the viscous
   residual targets the STABLE (viscosity-solution) object; ε=0.3 sits inside a two-sided
   stability window (ε=1.0 explodes).
2. **Raw-gradient StEik is a measured NO-GO at this state** (implementation-level per #307;
   paradigm intact): the paper's term |∇m^T H ∇m| carries a |∇m|² factor, and at the restored
   far-from-SDF state (|∇m| ≫ 1) the damping term is SELF-AMPLIFYING — eik_steik itself becomes
   a runaway term, monotone in W (575× → 1431× at 10× weight). The paper's regime is |∇u| ≈ 1
   near-SDF. Named follow-up (NOT built — the exact-term discipline): the NORMALIZED variant
   n^T H n, n = ∇m/|∇m|, removes the quartic scaling; build only if visco ever walls.
3. **The rollback guard eliminates the measured deadlock class** (0 skips across 120 steps;
   training continues; budget engaged) **but does not arrest THIS disease**: the runaway is a
   GRADUAL per-batch climb (~1.07×/batch) that a 5×-median spike detector cannot see — the
   median tracks the climb, the window stays "healthy", and the epoch-top refresh criterion
   (prev-epoch spike-frac) lets the last-good snapshot be OVERWRITTEN by mid-climb states (the
   ep129 rollback restored ep128 — already diseased — then lr×0.5 only delayed re-onset, exactly
   FEED-05q). Two named implementation fixes if the guard is ever promoted to a cure: (a)
   trigger on the per-term CANARY trend (eikonal trough-ratio > 2 over k epochs — the litsweep
   row-3 recommendation, which this build under-implemented as a median-multiple), (b) anchor
   snapshots to the best VERDICT epoch, not to spike-frac-healthy epochs.

## 5. Guard redesign — what changed and why it fights the disease

Legacy semantics (default, byte-identical): skip any batch > 5×frozen-median; median updates only
on ACCEPTED batches ⇒ a persistent loss-level shift = 100% skip forever (measured 3×: the
absorbing deadlock). Rollback semantics: the guard becomes an actuator that (i) lets bounded
oscillation THROUGH (the EoS self-stabilization feedback), (ii) on sustained runaway RETURNS the
run to the stable basin at a stable step size (restore + lr×0.5 + fresh median), (iii) spends a
bounded budget then reverts loudly to legacy. Snapshot policy: refreshed at each epoch top ONLY
when the previous epoch was healthy (spike-frac < trigger), so a runaway epoch can never
overwrite the good basin; snapshots are zero-copy (MLX arrays immutable; all consumers rebind).
KNOWN LIMITATION (documented): `lr_scale` + snapshot are NOT persisted in the resume sidecar — a
resume mid-rollback-regime restarts the scale at 1.0 (the spike_rollback rows log every cut;
acceptable for the probe-scoped actuator; persist before any n600 run that relies on it).

## 6. RECOMMENDED GO CONFIG (exact extra-trainer-flags delta vs the v4 argv)

**Primary (the measured-arm-matched config).** The winning arm ran the ORIGINAL v1/v3 lr
schedule (1e-3 cosine → ~9.78e-4 at ep101), i.e. the viscosity buys back the 2× step size the
v4 lr-cut spent. Exact delta vs the v4 argv:

```
REVERT   --lr 5e-4 --lr-end 5e-5        →   --lr 1e-3 --lr-end 1e-4     (the v1/v3 schedule)
ADD      --eikonal-viscosity 0.3
ADD      --eikonal-viscosity-anneal 1000
KEEP     --spike-guard-mode legacy       (default; do NOT ship rollback — measured non-cure §4.3;
                                          SC1' every-epoch skip alarm stays armed)
KEEP     everything else in the v4 argv unchanged (incl. moments-RESTORED resume, rewarmup flags)
```

Rationale: visco_03 is the measured arm AT this lr (120 steps, descending, best d_seg); the
ε-anneal (linear → 0 over the 1000-epoch schedule; ε ≈ 0.19 at tau@400, ≈ 0.08 in the finisher,
0 at end) is the ViscoReg vanishing-viscosity continuation — by the time ε is small the basin is
deep in the anneal where the schedule lr is also small. The ε=0 stencil-switch discontinuity
lands at the very end (ε already ~0).

**Conservative option B** (if the operator prefers belt+braces over speed): keep v4's
`--lr 5e-4 --lr-end 5e-5` AND add the two visco flags — two independent stabilizers; cost is the
2× slower step the diagnostic already priced. NOT the measured arm (ε=0.3 was probed at 1e-3,
not 5e-4); n24-verify first if chosen.

**Pre-registered n600 gates (unchanged from the diagnostic):** ep101-125 skip-rate <10%/epoch;
per-epoch `loss_terms` eikonal ≤ its ep101 restored level and non-inflating trough-to-current by
ep150; PLUS the new visco-specific gate: the `eik_stabilizer` ε-anneal rows present and the
"eikonal" term (now the viscous residual) DESCENDING through ep101-140 as in the arm.

## 7. Honest risks

1. **Smoothing-vs-sharpness**: any damping term risks eating the lane dashes (the MCF-erasure
   enemy). StEik is normal-direction-only by design; the arbitration's d_seg drift column is the
   MEASURED check — but at n24/40ep it is a short-horizon advisory read, not an n600 verdict.
2. **n24→n600 transfer**: the disease reproduces on the slice (control 24.2×), which is the
   transfer evidence we have; the n600 relaunch keeps SC1' armed every epoch regardless.
3. **λ_pre gap**: the 38/η constant failed here; lr policy should NOT be derived from
   38/λ_pre(t) until the ε-floor/minibatch/vintage suspects are separated.
4. **ViscoReg stencil switch at ε=0** is a small documented discontinuity; prefer ending the
   anneal DURING CE (before τ-descent) if visco rides the relaunch.
5. **Rollback + Muon finisher**: the lr cut reaches only the AdamW `opt.learning_rate`; the
   MultiOptimizer children own their LRs — rollback mode is NOT validated inside the finisher
   window (probe window is pre-finisher).

## Artifacts

- Build commit `89c2add13` (trainer + probe arms + λ runner + 27 tests).
- Identity A/B: `experiments/results/eik_stab_build_20260705/idref_{pre,post}` (+ logs, argv).
- λ probe: `experiments/results/eik_stab_build_20260705/lambda_pre/{probe.log, lambda_pre_probe_report.json}`.
- Arbitration: `experiments/results/eik_stab_arbitration_20260705/` (per-arm logs + argv +
  `report_control_baseline_v3.json` + `step_instability_probe_report.json`).
- Forensic set untouched: runs `015247Z/083453Z/095728Z/125950Z` + the sha256'd snapshot.
