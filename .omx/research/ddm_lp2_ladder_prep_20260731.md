# ddm_lp2 — LADDER PREP BUNDLE (P2/P3/P5) for the gc12 birth-completion ladder (2026-07-31, task #802)

**Model: claude-opus (Fable-class main routing).** Scorer-free prep bundle per gc12 §6 (P2/P3/P5) —
the rung-1 seal prerequisite + the (e1) rung-2 harness + the burn-4 terminal-deliverable skeleton. I
do NOT touch the scorer slot (ddm_qa92 owns it). **NOTHING launches.** This unit is MEANS: apparatus
that unblocks MAIN's fill-in fires, not a score mover.

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED.** Every artifact here is apparatus,
`score_claim=false`; the only numbers are the ε derivation (DERIVED) and the P2 self-test on real
telemetry ([macOS-CPU advisory], telemetry-only, no scorer).

## STORES-CONSULTED (recall-first, path+sha where pinned)
- gc12 `.omx/research/ddm_gc12_wall_branch_convocation_20260731.md` (b4d317538d) — §3 ladder (rung-1
  windows + rung-2 (e1) spec + terminal deliverable) · §5 seal demand · §6 P2/P3/P5 charters.
- fp1 `.omx/research/ddm_fp1_class_field_projection_20260731.md` — §4 QA91 super-nucleus inventory
  (schema + the birth-plateau-key-candidate verdict my P2 producer operationalizes).
- QA91 custody `/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/qa91_erased_lane.json`
  (schema `ddm_fp1_qa91_erased_lane.v1`; betti0_gt_lane 985 / super_nucleus_area_frac 0.9767 /
  nucleus_threshold_px 5 / birth_tail_slope 8.75). Analyzer code `experiments/ddm_fp1_qa91_erased_lane.py`.
- vh1 `.omx/research/ddm_vh1_v8v9v10_harvest_20260730.md` (2249f7955b) — rows 12/14 (birth corpus +
  birth_completion key + telemetry cadence) · row 13 (#208/#532 init verification) · row 7 (v9
  telemetry port = burn-4 PREREQUISITE) · §2 law taxonomy.
- Real burn telemetry `/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/telemetry.jsonl`
  (40 `topology_per_class` rows; exact schema: `betti0_gt`/`betti0_realized`/`gt_components_erased`/
  `smallest_surviving_gt_component_px`, 5-vectors, Lane=index 1).
- Trainer telemetry surface `experiments/train_tr1_partition_renderer_mlx.py:853` (`topology_per_class`
  def, 4-connectivity) → JSONL write `:1693`. Static coverage: `birth_completion` is a
  `resume_registry.DIRECT_CONTROLLER_NAMES` member (`src/tac/witness_control/resume_registry.py:118`,
  task #358) with an existing in-**witness**-trainer controller `witness_control/birth_completion.py`
  (part_frac/area-band criterion) — DISTINCT from my external tr1 window-loop plateau producer; I add
  NO prefix and touch NO registry.
- fd1/j2 solve surfaces: `ddm_family_d_gn_description.FamilyDGaussNewtonEngineV1.propose` (damped GN/CG)
  over `direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.loss_and_grad` (CE+hinge
  margin), and `coupled_margin_levelset.solve_active_set_kkt` (margin-through-R active-set KKT) — the
  REUSE targets P3 wires (does not fork).
- deferral ledger `.omx/research/ddm_deferral_queue_ledger_20260729.md` — QA24 (ΔS_rate ≈ 0.19–0.28,
  sg1 grid keep-mask) · QA84 (rowband D8/D16 pa1b theorem, `RowBandGrammar` built) · QA90 (temporal
  read) — the P5 arm-matrix receipts.
- pa1r `.omx/research/ddm_pa1r_pool_a_race_20260730.md` (fdb48e2c26) — control_tail parent custody +
  Pool-A-out verdict (the rung-1 parent lineage).

## THE THREE ARTIFACTS (what each is · its consumer · PLP/falsifier)

### P2 — the typed `birth_completion` gate-key PRODUCER (rung-1 seal prerequisite)
- **Code:** `src/tac/optimization/ddm_lp2_birth_completion.py` (core) + `tools/run_ddm_lp2_birth_completion_key.py`
  (CLI). **Tests:** `tests/test_ddm_lp2_birth_completion.py` (28 pass).
- **What it is:** the EXTERNAL evaluator the rung-1 continuation window loop calls BETWEEN windows
  (trainer stays sealed per gc12 §5). It fits the Lane `betti0_realized` slope over the last
  `window_gates` (default 5) checkpoints from `telemetry.jsonl`, vs the QA91 GT inventory, and emits a
  typed schema-versioned row (`ddm_lp2_birth_completion_key.v1`) with `gate_key="birth_completion"`.
- **Fire rule:** `fired ⇔ (slope ≤ ε) ∧ (above-nucleus erasure persists)`.
- **ε DERIVATION (provenance ladder — DERIVED, not hardcoded):**
  `ε = t_crit · max(SE_slope_ols, SE_slope_quant)` where
  `SE_slope_ols = sqrt((Σresid²/(W−2)) / S_xx)` (OLS slope standard error, window-data-driven),
  `SE_slope_quant = sqrt((1/12) / S_xx)` (integer-count rounding-noise FLOOR, uniform var 1/12),
  `S_xx = Σ(gate_i − mean_gate)²`, `t_crit` = one-sided Student-t critical value at α (dof = W−2).
  ε is the statistical band inside which the fitted birth slope is NOT significantly positive — i.e.
  the plateau test = "the slope is statistically indistinguishable from zero at confidence α." The OLS
  term ADAPTS to the real early-burn count churn (measured: 276→252→269→54); the quantization floor
  prevents a spuriously-tiny SE on a perfectly-linear-but-still-rising window from firing the key.
  **The ONLY non-derived scalar is α** (default 0.15866 = one-sided one-σ), custodied as a
  STATED-CONFIDENCE class-4 value with re-derivation trigger = "count-noise-model change or operator
  band choice" — it is the standard "slope not significantly positive" test's confidence, not a tuned
  magic number. `W=5` + `epochs_per_gate=10` are MEASURED-ANCHORS (fp1 5-gate slope; QA91 10-epoch
  telemetry cadence). "above-nucleus erasure persists" = `erased_count > 0 ∧ super_nucleus_area_frac
  > 0.5 ∧ above_nucleus_erased_estimate ≥ 1`, where `above_nucleus_erased_estimate = round(erased_count
  × super_nucleus_area_frac)` is a labeled DERIVED-ESTIMATE (the EXACT super-nucleus-erased count needs
  one scorer pass per the QA91 method_note — out of this $0 producer's scope).
- **Connectivity provenance (honest caveat for MAIN):** `betti0_realized`/`betti0_gt` are tr1 4-conn
  (trainer default `ndimage.label`); `super_nucleus_area_frac` is 8-conn GT (QA91). The persistence
  test mixes them as an ESTIMATE — already labeled DERIVED-ESTIMATE. Both `betti0_gt_lane` sources
  agree at 985 (the producer validates telemetry `betti0_gt[Lane] == inventory.betti0_gt_lane`,
  fail-closed on mismatch — catches a wrong class index or a schema reorder).
- **Consumer:** the gc12 §5 rung-1 window loop (external, between windows) — the STOP signal that ends
  window extension when the birth wall is hit. gc12's "preferred successor form" (typed in-trainer key)
  is future work; this session's trainer is sealed, so the external producer is the correct vehicle.
- **PLP:** apparatus — no score prediction. The producer's job is to make the plateau decision typed +
  fail-closed + deterministic. **Self-test on the REAL bc1 endpoint telemetry: `fired=False, slope=7.60
  comp/gate, ε=3.34, erased=509, above_nucleus_est=497`** — CORRECT: births still rising at ep399
  (burn ended pre-plateau, matching QA91 birth_tail_slope 8.75), so the window loop keeps extending.
  A synthetic plateau window fires True. **Fail-closed:** malformed/missing telemetry, insufficient
  window (<3 pts), duplicate epochs, degenerate S_xx, realized>gt all raise `BirthCompletionTelemetryError`.

### P3 — the (e1) solve-seeded-births HARNESS (build-only; does NOT run)
- **Code:** `src/tac/optimization/ddm_lp2_e1_seeding_harness.py`. **Tests:**
  `tests/test_ddm_lp2_e1_seeding_harness.py` (21 pass).
- **What it is:** the build-only harness for gc12 rung-2 arm (e1). Per still-erased super-nucleus Lane
  component: (1) `extract_covering_tokens` (real deterministic pixel→token-cell geometry, LOCAL +
  optional 1-ring dilation), (2) a BOUNDED LOCAL token solve via an INJECTED `LocalTokenSolver`
  Protocol — the real fd1/j2 providers are `FamilyDGaussNewtonEngineV1.propose` /
  `coupled_margin_levelset.solve_active_set_kkt`, active-set restricted to the covering tokens (REUSE,
  not fork), (3) record the per-component `solve_residual` = local-token REACHABILITY (rg3 zero-support
  analog), (4) `verify_seed_init` = #208 rare-class-protected (Lane channel live) + #532 rendered-init
  verification (per-class mass vs GT priors) BEFORE marking for reconcile, (5) `assemble_seeding_report`
  (typed `ddm_lp2_e1_seeding_harness.v1`) marking ONLY init-verified accepted seeds.
- **Scorer-free-buildable:** the scorer (`HardOracle`) and the GN/CG solver are INJECTED (b2b stub
  pattern); `StubLocalTokenSolver` + `stub_hard_oracle` exercise all plumbing without a real scorer or
  a real solve. MAIN wires the real fd1/j2 + frozen CPU-torch SegNet + real endpoint components at fire.
- **PREREGISTERED FALSIFIER (carried in code + docstring):** `evaluate_survival_falsifier(seeded_ΔS,
  survived_ΔS)` — survival fraction < 0.50 (gc12 §3 rung-2 (e1) ANCHOR) ⇒ (e1) CLOSES at FORMULATION
  scope (local-solve-seed on this vehicle); the erasure force re-erased the seeds. MAIN calls it AFTER
  a real reconcile tail runs (build-only now: I built the evaluator + schema, it does not run the tail).
- **Consumer:** gc12 rung-2 routing (fires when O ≥ 0.25 ∧ F < 0.7·O per ddm_qa92) → burn-4 seg axis
  if (e1) wins the rung-2 race.
- **Scope checks (gc12 §2):** stays ON the render manifold (fp1's receiver tax inapplicable — re-renders
  from seeds, does not composite a flat field); SPENDS token bytes (pa1r-favorable); NOT nv1's null-snap
  (opposite sign). **PLP:** the burn birthed 476/985 unaided ⇒ tokens CAN express lane components;
  genuinely open = (i) local-token reachability of the residual set (the `solve_residual` measures it),
  (ii) survival through the reconcile tail (the falsifier measures it) — both UNMEASURED until MAIN runs.

### P5 — the burn-4 CHARTER SKELETON (receipt-parameterized template)
- **Doc:** `.omx/research/ddm_burn4_charter_skeleton_20260731.md`.
- **What it is:** the fill-in template for gc12's terminal deliverable — the from-birth composed burn
  where the RATE axis rejoins. §2 arm matrix (QA24 granularity re-race [ΔS_rate ≈ 0.19–0.28] × rung-2
  birth-completion winner × QA84 rowband), each cell with its DERIVED band + falsifier + `⟪UNKNOWN:…⟫`
  slots filled from rung-0/1/2 receipts. §3 names the **v9 telemetry port to TR1 as the single hardest
  PREREQUISITE** (vh1 row 7: per-term #304 + term-domination/inert #321 + liveness + positive-control
  #404 — the guards are only as good as the telemetry that trips them). §4 sealed-config demand mirrors
  gc12 §5. §5 composed arithmetic vs 0.172141 with fill-in slots + the means-vs-ends firewall line. §6
  PLP lines on every proposed burn-4 measurement.
- **Consumer:** MAIN charters burn-4 by filling the `⟪UNKNOWN⟫` slots on the rung-2 receipts — the
  skeleton makes that a fill-in, not a design session (gc12 op-r 4 / §6 P5).

## NOT BUILT (gated, per charter)
- **P4 (carrier design doc)** — gated on ddm_qa92's O/F verdict; NOT built (correctly held).
- **Nothing launched.** No scorer touched (ddm_qa92 owns the scorer slot). No parallel-session WIP
  touched (direct_description_carrier_compose.py, ddm_qa43_two_plane_parallax_probe.py, burn_out).

## verdict_scope ledger
- P2 ε derivation: DERIVED (OLS + integer-count floor); α = STATED-CONFIDENCE. Self-test on real
  telemetry: MEASURED (telemetry-only, no scorer) — correctly not-fired at the pre-plateau endpoint.
- P3: apparatus (build-only); the (e1) falsifier is preregistered at FORMULATION scope; no run, no claim.
- P5: apparatus (fill-in template); all bands DERIVED with `⟪UNKNOWN⟫` slots; no score claim.
- No prior negative re-opened; no registry/prefix added; no trainer edit.

**Pointer 0.1910828242 [contest-CPU] UNMOVED.** [no-triality] [p0-ledger-ok]
