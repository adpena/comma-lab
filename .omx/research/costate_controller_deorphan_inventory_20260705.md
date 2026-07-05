# Costate controller DE-ORPHAN INVENTORY — task #303 Phase A (2026-07-05)

**Purpose.** Per the #247 velocity-driven-orphaning finding (memory
`velocity_driven_orphaning_the_deepest_signal_loss_meta_bug_20260703`) and the meta-layer
memory (`project_meta_layer_above_triality_hamiltonian_control_costate_20260703`): before
building the costate controller, inventory EVERY existing observer/control/recommendation
surface and decide, per surface, whether the controller CONSUMES it, WRAPS it, SUPERSEDES
it, or leaves it OUT-OF-SCOPE. The controller (`src/tac/witness_control/`) is the ONE
canonical consumer for witness-campaign control signal from this landing forward.

**Verdict counts: CONSUME 6 · WRAP 3 · SUPERSEDE 1 (scoped) · OUT-OF-SCOPE 1.**

| # | Surface | Producer/consumer status (verified by grep, 2026-07-05) | Decision |
|---|---------|-----------------------------------------------------------|----------|
| 1 | `tools/witness_control_monitor.py` (#289) | LIVE, thinly consumed: reads verdict rows; sole production importer was `tools/dashboard_control_telemetry.py`. Its `classify_trajectory` is the CANONICAL classification spec (the trainer replicates it, parity-guarded by `experiments/test_closed_loop_control.py`). | **CONSUME** — the shadow controller imports it unforked (`shadow_controller._classify`); it now has a second production consumer. |
| 2 | Trainer `--closed-loop-control` (build-3, `experiments/train_levelset_witness_realized_through_R_mlx.py` ~L1250-1345, 3615-3800, 4339-4345) | LIVE default-OFF. Observes: captured verdict rows (decide-on-previous M2 reorder). Actuates: BOUNDED eikonal bump (`eff=min(sched+bump, max(cl_max, sched))`) + early-stop arming ONLY. Emits `{"stage":"closed_loop"}` rows. | **CONSUME** (Phase A: the shadow controller parses its `closed_loop` rows via `dashboard_control_telemetry.parse_stage_rows`). Phase B: it remains the ONLY in-run actuator; the costate controller extends its action set solely via new default-OFF flags decided at launch — never a parallel in-run actuator. |
| 3 | `tac.witness_dsl` (curriculum_dsl / campaign / gauge / powerplay) | LIVE as compile/validate library (consumers: triality_drift_detector, dashboard_server, preflight, canonical_equations). `compile_trainer_argv`/`compile_daemon_argv` have NO production launcher caller — the argv-emission path is ORPHANED from launch. | **WRAP** (Phase B design): the controller's actuation surface IS DSL argv emission — u* compiles to a validated flag-diff via `compile_trainer_argv`, GO-gated, operator-launched. De-orphans the emission path. NOT touched in Phase A (gauge.py is sibling #302 territory; `ControllerGauge` integration = named follow-up after #302 lands). |
| 4 | Cathedral autopilot (`tools/cathedral_autopilot*.py`, `src/tac/cathedral*`) | LIVE for the compression-substrate campaign (many register_* consumers; last touched 2026-06-06). ZERO references to levelset/witness — orphaned RELATIVE to this campaign. It was the early, sprawled costate-controller attempt (the meta-layer memory's diagnosis: sprawled because it lacked the variational skeleton). | **SUPERSEDE (scoped to the witness campaign)**: witness-campaign control recommendations route to `costate_shadow.jsonl` + the shadow controller, NOT to a new cathedral consumer. Migration name: any future witness-side cathedral consumer registration is refused in favor of `tac.witness_control`; the cathedral stack is NOT deleted or modified (it remains the compression-substrate recommender). |
| 5 | `tools/system_memory_governor.py` | LIVE actuation authority (admission gate + throttle + bands + fail-closed reconcile). Consumed by launch_witness_run, spawn_durable_daemon, memory_guard. | **CONSUME** — architecture patterns stolen as prescribed (registry-anchored accounting → `RunInputs`; bands → costate ± bands; fail-closed → UNIDENTIFIABLE refusal; pure-decision-fns-over-snapshot → `build_shadow_report`). Phase B: launch admission REMAINS governor-owned; the controller never acquires it. |
| 6 | Per-stage attribution tools (#253/#255): `tools/erasure_timing_attribution.py`, `tools/witness_per_stage_annulus_attribution.py` | LIVE standalone CLIs, NO programmatic consumer. They are the per-class marginal-ΔS (per-class λ) PRODUCERS the aggregate verdict rows cannot supply. | **WRAP** (named follow-up + probe queue): per-class costates are UNIDENTIFIABLE from verdict rows alone; the follow-up wires these tools' per-class outputs into `costate_estimator` as evidence rows. Until then the controller honestly reports per-class λ as a gap. |
| 7 | Focal γ calibration (`experiments/probe_focal_gamma_calibration.py`, memo `focal_boundary_calibration_20260705.md`, eq `focal_gradient_concentration_v1`) | LIVE probe + registered equation. IS the exemplar measured ∂(internal-state)/∂(knob) costate probe (γ-sweep → grad-share). | **CONSUME** — generalized as `costate_estimator.sweep_finite_difference` (status PARTIAL: internal observable, chain to S unmeasured — exactly the focal memo's own honesty). |
| 8 | `tools/launch_witness_run.py` + `src/tac/witness_autoconfig.py` | LIVE; consume ONLY the governor's read helpers. No monitor/controller coupling today. | **WRAP** (Phase B insertion point, GO-gated): the launcher optionally reads the latest `costate_shadow.jsonl` recommendation as a pre-launch advisory display. Never auto-applied. |
| 9 | `witness_control*/costate*/shadow_control*` collisions | NONE exist beyond the monitor (verified find). | n/a — `src/tac/witness_control/` is collision-free. |
| 10 | `tools/render_levelset_dashboard.py` + `tools/dashboard_control_telemetry.py` parsers | LIVE. `_parse_verdicts` (L128) / `_parse_launch_sh_flags` (L253) / `parse_stage_rows` are the parsing conventions. | **CONSUME** — `parse_stage_rows` + monitor `_read_verdicts` imported directly; `_parse_launch_sh_flags` replicated verbatim-contract (the dashboard module imports matplotlib at import time) with a PARITY regression test on the real #205 launch.sh (`test_parse_launch_flags_parity_with_dashboard`). |

## What this discharges

The #247 spirit: the passive-DSL gap ("a costate producer with no controller consuming
it") is closed for the witness campaign by ONE canonical consumer — the shadow controller
now consumes the monitor (1), the trainer's closed-loop telemetry (2), the dashboard
parsers (10), and the probe pattern (7); it names the wraps (3, 6, 8) as Phase-B/follow-up
integrations rather than rebuilding them; and it scopes the cathedral (4) out of the
witness path explicitly instead of silently duplicating it.

## Cross-refs

- Sibling study: `.omx/research/council_grand_symposium_curriculum_derivation_20260705.md`
  §B.4 (hybrid Bolza frame; costate = switching function) + §C.d (#247 target design) —
  its derived τ-path/hand-off laws drop into Phase B as REFERENCE TRAJECTORIES (see the
  design memo's interface stub). Not blocked on.
- Design memo: `.omx/research/costate_controller_design_20260705.md` (T2).
- Code: `src/tac/witness_control/{costate_estimator,shadow_controller}.py`,
  `tools/costate_shadow_report.py`, tests `src/tac/tests/test_witness_control_costate.py`.
- Canonical equation: `costate_lambda_marginal_ds_v1` (registered 2026-07-05).

Axis: everything here is [macOS advisory] NON-PROMOTABLE; pointer 0.19110 UNMOVED (means).
