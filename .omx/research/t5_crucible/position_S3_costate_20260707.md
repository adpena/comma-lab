# T5 CRUCIBLE — SEAT S3 POSITION — CONTROL/COSTATE (Pontryagin/Rudin charter)

Seat: S3 · 2026-07-07 · anti-anchoring honored (no other position_S*.md read).
Axis discipline: every number below is [macOS-CPU/MLX advisory] NON-PROMOTABLE unless
tagged otherwise. Pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS.

**Headline (measured today, $0, inline):** the pinned D-3/4/5 first measurement — HVP-Lanczos
on the saved #205 clean-baseline checkpoint — was RUN, not just spec'd. At **ep650 EMA-best**
(the run's best point, frozen-anneal state read from the ckpt itself: softmax_temp 0.3098,
hosc_beta 2.9489), the gradient-seeded Lanczos tridiagonal of the tau-stage training loss is
**strongly INDEFINITE**: Ritz values after 7 iterations
`[−369.7, −94.1, −18.5, +14.6, +61.5, +93.9, +139.3]` (K=8 subset, seed 0 — the same subset
#341 solved). The most-negative reachable curvature (−369.7, stable from iteration 3 onward)
is **2.65× larger in magnitude than λ_max (+139.3)**. Consequences, each labeled below: the
TerminalSolve in-basin condition was NOT met at the run's best point; the 1st-order plateau
narrative ("tau stage exhausted") is NOT established — second-order descent structure existed
that `powerlaw_meat_exit` is constitutionally blind to; and the cold ep726 Muon switch
(quenched +27.5%, never re-beat ep650) acted on a curvature-blind state.

---

## 1. Position

### 1.1 SENSE — the full signal inventory (existing = verified wired; NEW = the GN/Fisher spectrum)

**Existing producers the controller already consumes** (all verified in source; de-orphan, don't
rebuild):

| Signal | Surface | Status |
|---|---|---|
| λ ladder (ANALYTIC λ_seg=100 · λ_pose=5/√(10·d_pose) · λ_bytes=6.659e-7 S/byte; MEASURED per-stage dS/dep windowed-OLS ± stderr; stage-jump; rollback gain) | `tac.witness_control.costate_estimator` | wired |
| Trajectory classification + recommendation set (POWERPLAY never-regress at the recommendation layer) | `tac.witness_control.shadow_controller.build_shadow_report` / `write_shadow_row`; classification via `witness_control_monitor.classify_trajectory` | wired; observer auto-start per governed launch |
| 1st-order predictive exit: AIC power-law vs exponential tail, extrapolated remaining meat | `tac.witness_control.powerlaw_exit.powerlaw_meat_exit` → `producer_bridge._powerlaw_exit_signal` (`powerlaw_exit.stage_tail_fit`) | wired; **curvature-blind by construction** |
| Annulus telemetry #333 (~97% d_seg mass in ~4.7%-area annulus) | run-dir `annulus_live.jsonl`; `tools/costate_digest.py::section_annulus` | wired |
| Per-class verdict/λ sensors | `tac.witness_control.perclass_verdict` (#315 family) | wired (per-class λ still the top probe-queue gap per the design memo §2) |
| Liveness / skip-rate / spike alarms (confound L1) | trainer `SpikeGuardRollback` + liveness stamps | trainer-side, wired |
| Duty-to-measure activation ledger + failure ledger + sensitivity map + master-gradient + autopilot ranker | `producer_bridge.read_producer_signals` (5 producers, `producer_bridge.py:73–260`) | wired |
| Event-curriculum readiness rows (#315): plateau rel-eps 1e-4 / window 25 / min-stage 250; nucleus predicate π₁=w/σ≳5 | levelset trainer `_evt_resolve_seg_form` / `_evt_readiness_row` | wired end-to-end, fixed epochs = CAPS |

**SENSE honesty item (apparatus gap, binding on the crucible):** the activation ledger is NOT
wired to real-run activations (grounding packet caveat). Per-lever activation ground truth =
`launch.sh` in the run dir. Named fix: `launch_witness_run` should ingest the compiled argv
into `tac.witness_dsl.activation_ledger` at launch time (one write, score-neutral). Until
then, DECIDE reads run configs, not the ledger, for "did it fire."

**NEW SENSE state — the GN/Fisher second-order SPECTRUM (the pinned ONE live deep-math lever).**
Design and first measurement:

- **What:** top/bottom Ritz values of the Hessian of the stage training loss at saved
  checkpoints, from gradient-seeded Lanczos with full reorthogonalization. Gradient seeding
  makes the Krylov space the *reachable* curvature (what descent can actually use) and gives
  the Newton decrement ½·‖g‖²·(e1ᵀT_k⁻¹e1) for free when T_k is PD — the mechanistic
  "remaining descent in the local quadratic model," i.e. the curvature-aware replacement for
  the phenomenological remaining-meat extrapolation.
- **HVP implementation:** REUSED, not rebuilt — `tools/quadratic_basin_finisher_probe.py`
  (#341) `SolveCtx.hvp_pair` = `mx.vjp` of `mx.grad` (exact symmetric HVP; MLX
  forward-over-reverse jvp fails: SliceUpdate JVP unimplemented — documented in that file).
  Its `Ctx` also solves the two hard problems already: self-orient dir-feats fixed-point
  reconstruction (`feats_state_main_gt1.npz`, full-P) and the checkpoint's OWN frozen schedule
  point (softmax_temp / hosc_beta read from ckpt cfg, never args) — which is exactly the
  truncated-anneal conditioning the mid-crucible correction requires (β frozen 3.177 / τ 0.216
  at ep726; my ep650 probe correctly ran at the ep650 state 0.3098/2.9489).
- **Probe harness landed this seat:** `experiments/t5_s3_hvp_lanczos_probe.py` — chunked
  resumable foreground (per-iteration state persist under
  `experiments/results/t5_s3_hvp_lanczos_20260707/`), RAM-floor gated, deterministic
  (mx.cpu + seeded subset), `--timing-only` feasibility gate.
- **Which checkpoint:** primary = ep650 EMA-best (`basin_finisher_probe_20260707/
  ema_best_snapshot.npz`, __epoch=650, the run's best d_seg 0.0033662). Per the
  truncated-anneal correction: ep1000 is a cold-quench artifact state — useful as a
  *contrast*, not as a converged finisher; ep726 (MuonStart) reads the pre-switch state.
  RECESS extends to ep726/ep1000 + the LIVE (non-EMA) weights from
  `levelset_resume_stageTau_muon_*.npz` (EMA shadow ≠ optimizer iterate; both matter).
- **How many Lanczos iterations:** extreme Ritz values converge in 3–5 iterations (measured:
  λ_- stable at −369.7 from iter 3); k=12–32 for interior structure + decrement. Full
  reorthogonalization is free at this scale (n_dim=111,095; V at k=32 ≈ 14 MB).
- **Runtime/memory budget (MEASURED today):** mx.cpu — grad 5.1 s/pair, HVP 10.9 s/pair;
  K=8 → ~87 s/iteration; peak RSS well under the 8 GiB envelope (lstars ~0.9 GB + feats).
  Full-P=600 per HVP ≈ 1.8 h on CPU → RECESS with the K-ladder design (§4-R1), or MLX-GPU
  throughput (never a bit-exact proof surface — memory L70) with CPU spot-verification.
- **Eigen-summary that feeds DECIDE** (the SENSE→DECIDE contract): per checkpoint/stage,
  the tuple `{lambda_max, lambda_min_ritz, n_negative_ritz, newton_decrement (PD case),
  grad_norm, k_pairs, units}` — persisted as JSON, surfaced as a new
  `ProducerSignal(name="gn_spectrum.checkpoint_lanczos")` in `producer_bridge` (mirror of
  `_powerlaw_exit_signal`; honest `available=False` when no spectrum artifact exists).

**MEASURED first reading (the deliverable of the $0 probe):** at ep650-EMA, K=8/seed-0,
7 Lanczos iterations: Ritz `[−369.7, −94.1, −18.5, +14.6, +61.5, +93.9, +139.3]`;
`grad_norm 0.787` (surrogate units — the EMA shadow is not a stationary point);
`n_negative_ritz 3/7`; Newton decrement undefined (T indefinite). Units: the SOLVE loss
`100·tau_softplus(τ=0.3) + 0.001·length` — NOT d_seg; the surrogate↔d_seg map is a separate
calibration (§4-R3). Subset caveat: K=8 is the #341 solve subset; #341 itself measured the
subset-solve gap (K=8 overfits +5.1% net) — eigenvalue ESTIMATION has a different failure
mode (E[H_K]=H_600, variance ~1/K) but the full-P confirmation is owed before any launch
decision leans on the magnitude (§4-R1 pre-registered bands + kill thresholds).

### 1.2 DECIDE — λ marginal-ΔS arbitration, now curvature-aware

- **The arbitration spine stays:** rank candidate actions by expected ΔS per cost via the
  costate ladder (λ_seg=100 exactly; λ_pose=5/√(10·d_pose) — pose/seg crossover at
  d_pose≈2.5e-4 falls out; λ_bytes≈6.659e-7 S/byte anchors every byte-costed lever), with
  propagated stderr bands and the POWERPLAY never-regress refusal (central predicted ΔS ≥ 0
  ⇒ refused) already implemented at the recommendation layer.
- **What the 2nd-order state changes (the ep450-miss fix):** the ep450 λ-backtest miss
  (predicted +0.0060 [0.0018,0.0103] vs realized +0.0004) was linear λ-extrapolation
  over-predicting under decay. The spectrum replaces the guessed decay law with a mechanistic
  one: in the local quadratic model the loss tail is an exponential MIXTURE with rates set by
  the eigenvalues (c_i·e^(−2λ_i·η_eff·t)), so the horizon forecast becomes
  spectrum-calibrated instead of a phenomenological power-law/exponential AIC coin-flip.
  Division of labor: `powerlaw_meat_exit` stays the cheap every-verdict in-run fit; the
  spectrum probe runs at per-stage checkpoint boundaries ($0, out-of-process) and
  RECALIBRATES the exit's horizon + floor.
- **Stage-exit rule (the curvature-aware exhaustion test):** declare a stage exhausted only
  when ALL of: (a) 1st-order remaining_meat < floor (existing); (b) gradient-Krylov T_k is PD
  and Newton decrement < floor (quadratic-basin descent gone); (c) no usable negative
  curvature: |λ_-|·Δ²/2 < floor at the trust radius Δ. Today's measurement shows (b)/(c)
  FAIL at ep650 — i.e. the clean baseline's best point was NOT exhausted in the 2nd-order
  sense; what followed (cold Muon quench +27.5%) was a curvature-blind response. [INFERRED
  at K=8 — the full-P RECESS either confirms or kills this.]
- **TerminalSolve gating (#341):** Newton/CG-class finishers are legal only under (b) — T_k
  PD in the gradient Krylov space. Until then only damped-LM steps (ρ-gated `lm_accept`,
  already in #341) are admissible. The measured LM ρ 0.847/0.868 with damping + today's
  indefiniteness are CONSISTENT: the quadratic model is locally valid along damped steps
  while the undamped Newton system is indefinite. Subset-solve NO-GO (#341, +5.1% net
  overfit) stands: any finisher solve is full-P in-trainer only.
- **Trust-region + monotone improvement (the TOP-D transferable principle):** every emitted
  step/config carries a bounded-step discipline — max accepted step sized by the current
  λ_max (½·λ_max·‖Δθ‖² ≤ accepted surrogate-risk), acceptance only on measured improvement,
  rollback-to-best as the enforcement (per-stage EMA checkpoints are the non-negotiable
  substrate). This also gives the *principled* stage-transition rewarmup: rewarmup until
  effective step·λ_max is inside the stability edge, replacing the ASSUMED fixed 8–20 ep
  shape (keep current values run-1; measure at boundaries).
- **Lever-queue ranking:** duty-to-measure levers ranked by |expected ΔS|/cost with the
  diversity floor f=1/4 reserved for never-fired levers (Weng invariant ii, binding in the
  design memo appendix); per-lever activation ground truth from launch.sh until the ledger
  ingest lands (§1.1).

### 1.3 ACT — autonomous vs operator-GO (CONTAINMENT binding)

**Autonomous (advisory only, structurally incapable of process control):** shadow rows +
recommendation sets; lever-queue/duty-to-measure ranking; $0 spectrum probes on saved
checkpoints at stage boundaries (read-only, score-neutral observability → defaults ON per
the default-off-orphan rule); event-curriculum CONDITION INPUTS; RECOMMENDED-CONFIG file
emission — always compiled + validated through
`tac.witness_dsl.curriculum_dsl.compile_trainer_argv` / `.validate()`, never raw flag edits.
The Weng authority-outside-the-loop invariant binds: no emitted recommendation may target
evaluator/permission surfaces (the DSL vocabulary contains none — keep it that way).

**Operator-GO (never autonomous):** heavy/paid launches, run stops, live-config changes.
In-run actuation remains the trainer's build-3 bounded loop exclusively; extensions only as
default-OFF flags, byte-identical when OFF.

**How ACT programmatically reshapes the curriculum (event-exit predicates through the DSL):**

1. **`ExitEvent(criterion="powerlaw_meat")`** — gap-kind criterion (named in
   `powerlaw_exit.py`'s docstring as the intended consumer); conservative compile = fixed
   boundary CAP + TrainerSupportGap. Run-1 needs no trainer edit: the wired #315
   event-trigger (plateau + nucleus predicate) carries the in-run exits; powerlaw feeds the
   controller's advisory horizon.
2. **NEW `ExitEvent(criterion="quadratic_basin")`** — the Muon/finisher fire condition: fire
   the warm-started finisher (`MuonWarmStart`, #269/#270 built) when the per-stage-checkpoint
   spectrum says basin-entered (T_k PD ∧ decrement < floor), with the fixed epoch as CAP
   (byte-identical when unfired — same semantics as #315). This is the derived replacement
   for `muon-start-epoch 726` (PR95 clock cargo); mod32cap's cold ep726 fire is the measured
   counterexample. Run-1 realization WITHOUT new trainer machinery: probe runs out-of-process
   on the per-stage checkpoints (saved every 25 ep); controller emits the advisory
   fire/hold; full in-trainer support is a named build item, not a run-1 gate.
3. **Margin-GATED island support (folded from the treatment-arm memo,
   `council_t3_symposium_islands_treatment_arm_20260706.md`):** amplification support
   Ω = {pixels with big-3 margin preserved} makes island birth net-positive BY CONSTRUCTION
   (Δd_seg ∝ n_isl − n_big3; gating drives n_big3→0). ACT emits the gate as a curriculum
   condition input; the wired cheap form is the #300 `--witness-alone-island-loss` soft-gate
   (MEASURED absorbing: lane within-flip −45% while total d_seg descended); the HARD
   costate-gated arm stays DESIGNED-ONLY pending #268 exact-S_R (msal_uni reachability proxy
   measured AT CHANCE — do not gate on it). In the low-lane-share regime the cheapest lever
   is `AnalyticLaneRenderBand` (#213, representation win) — the controller must not spend a
   training-arm budget where a render-band pays the same ΔS at zero d_seg cost.

### 1.4 Wiring — de-orphan, don't rebuild (exact surfaces verified)

- `src/tac/witness_control/producer_bridge.py`: add `_gn_spectrum_signal()` →
  `ProducerSignal("gn_spectrum.checkpoint_lanczos")`; wire into `read_producer_signals`
  (list at ~line 258 alongside the 5 existing producers).
- `src/tac/witness_control/shadow_controller.py`: `_recommendations` gains two advisory
  candidates consuming the spectrum row: `HOLD_STAGE_NEGATIVE_CURVATURE` (refuse
  exhaustion/advance when |λ_-| usable) and `FIRE_FINISHER_BASIN_ENTERED`; both flow through
  the existing never-regress refusal.
- `tools/costate_digest.py`: one fail-open section (pattern of `section_annulus`) rendering
  `{λ_max, λ_-, n_neg, decrement, age, k_pairs}`.
- `tac.witness_dsl.curriculum_dsl`: the two gap-kind `ExitEvent` criteria (conservative
  compile) + a `GNSpectrumProbe` observability lever (default-ON, score-neutral, checkpoint-
  boundary cadence) as a DSL `Lever` factory so it is HELD, not orphaned.
- `tools/launch_witness_run.py`: activation-ledger ingest of the compiled argv (closes the
  ledger-semantics gap the grounding packet flags).
- Probe: `experiments/t5_s3_hvp_lanczos_probe.py` (landed) reusing
  `tools/quadratic_basin_finisher_probe.py::{Ctx, SolveCtx.hvp_pair, choose_subset_pairs,
  ram_floor_ok, flatten_masked, unflatten_masked}`.

---

## 2. Derivations + assumption tags (#363)

- HVP = mx.vjp∘mx.grad exact for symmetric H; jvp unavailable (SliceUpdate) —
  VERIFIED-VIA-SOURCE(`tools/quadratic_basin_finisher_probe.py:14–15,433–447`).
- ep650 EMA-best d_seg 0.0033662, __epoch=650 — VERIFIED-VIA-ANCHOR(`levelset_best.json` +
  `ema_best_snapshot.npz` cfg).
- Checkpoint frozen schedule point temp 0.3098 / β 2.9489 — VERIFIED-VIA-ANCHOR (npz cfg
  read inline today); consistent with the truncated-anneal correction (freeze at ep726 =
  0.216/3.177 — coordinator-supplied MEASURED fact, sibling-seat n600 probe).
- Ritz spectrum at ep650 (K=8, 7 iters) `[−369.7 … +139.3]`, grad 5.1 s/pair, HVP
  10.9 s/pair — MEASURED today, artifacts
  `experiments/results/t5_s3_hvp_lanczos_20260707/spectrum_ep650_K8_s0.json` +
  `lanczos_state_ep650_K8_s0.npz` ([macOS-CPU advisory], deterministic seeded).
- "ep650 was not 2nd-order exhausted; cold-Muon was curvature-blind" — INFERRED (from the
  K=8 spectrum + the mod32cap trajectory; K=8→full-P transfer is the load-bearing
  assumption ⇒ RECESS-R1 gates it; verdict PROVISIONAL until then).
- Newton-decrement-from-tridiagonal (½‖g‖²·e1ᵀT⁻¹e1, gradient-seeded) — DERIVED (standard
  Lanczos-quadrature identity; valid only for PD T_k, enforced in code).
- λ_bytes 6.659e-7 S/byte; ep450 λ-backtest miss = linear extrapolation under decay —
  VERIFIED-VIA-ANCHOR(`costate_controller_design_20260705.md` §1–3; pinned in convening §2).
- Subset-solve NO-GO +5.1% / LM ρ 0.847/0.868 — VERIFIED-VIA-ANCHOR(#341 landing, memory
  L77, eq `quadratic_head_chart_subset_solve_gap_v1`).
- Muon −32% vs AdamW; cold-Muon quench +27.5% never re-beating ep650 — VERIFIED-VIA-ANCHOR
  (fork A/B 2026-06-22; coordinator-supplied sibling n600 probe on the mod32cap log).
- Margin-gated island support net-positive-by-construction; #300 soft-gate absorbing —
  VERIFIED-VIA-ANCHOR(`council_t3_symposium_islands_treatment_arm_20260706.md` Lens A).
- mod32cap = clean unconfounded T3 baseline (no seeding/lane-prior/band/island levers;
  eik 0) — VERIFIED-VIA-ANCHOR(`council_symposium_clean_config_20260705.md` per operator
  correction; `launch.sh` cross-checked inline: `--eikonal-weight 0`, no seed/band flags).
- EMA shadow ≠ optimizer iterate for spectrum purposes — ASSUMED-unavoidable for run-ended
  probing (live weights exist in the resume sidecar; RECESS-R1 measures both).

## 3. PR95 cargo-cult audit (my face)

| Element | Verdict | Basis |
|---|---|---|
| `powerlaw_meat_exit` horizon/floor exit | DERIVED-FROM-WITNESS-MATH (weak-KAM O(1/t) tail on the binding lane class) — KEEP, but curvature-blind: complement, never sole exhaustion authority | `powerlaw_exit.py` docstring derivation |
| `--muon-start-epoch 726` fixed clock | DROP/REPLACE → `ExitEvent(criterion="quadratic_basin")` with 726→CAP fallback | PR95 one-trajectory clock transfer; mod32cap cold-fire measured counterexample; today's ep650 spectrum |
| Muon itself (finisher) | JUSTIFIED-KEPT (−32% vs AdamW MEASURED) — warm-started + LR-annealed (#269/#270) | fork A/B |
| lr 1e-3→1e-4 cosine | JUSTIFIED-KEPT-PENDING-MEASUREMENT — the "sharpness probe before churning" (#302 row 5) IS this seat's spectrum instrument; λ_max-vs-stage sets the ceiling principledly | today's probe = the instrument |
| Rewarmup 8–20 ep fixed shape/floor | JUSTIFIED-KEPT run-1; replace length by the stability-edge rule (step·λ_max) once boundary spectra exist | #302 row 17 (shape ASSUMED) |
| EMA 0.997 as verdict state | JUSTIFIED-KEPT for score; for SENSE, spectrum must eventually be read at BOTH EMA and live weights (π-group violation note stands for the finisher A/B) | #302 B.6 |
| Aggregate-verdict-only costate state | CARGO (named in the design memo's own Assumption-Adversary row) — per-class λ via #253/#255 stays the top SENSE gap | design memo §2 |

## 4. RECESS measurement proposals

**R1 — Full-P spectrum ladder (the gate on today's headline).** What: gradient-seeded
HVP-Lanczos k=16 at K∈{8,32,128} (seeds 0,1) + full-P=600 if budget allows, at ep650-EMA,
ep650-live (resume sidecar), ep726-MuonStart, ep1000; each at its own frozen ckpt-cfg point.
Command: `experiments/t5_s3_hvp_lanczos_probe.py --ckpt <…> --tag <…> --k-pairs K --iters 16
--max-seconds 150` looped (chunked-resumable; governed foreground). Cost: K=32/k=16 ≈ 1.6 h
CPU per checkpoint (or ~6 min on MLX-GPU throughput, CPU spot-verified); full-P on GPU
≈ 3–4 h. Mem < 8 GiB. Pre-registered bands: (i) ep650 indefiniteness PERSISTS at K=128
(|λ_-|/λ_max > 0.5); KILL the "not-exhausted" verdict if it shrinks < 0.1 → then the wall is
capacity/basis and Arm A is the only mover (either outcome is decision-grade for the
crucible). (ii) ep1000 (cold-quench artifact) shows λ_max ≥ 1.5× ep650's (sharper quenched
state — the "why Muon didn't pay" signature). (iii) ep650 PD-part decrement ≥ the realized
ep650→ep1000 loss gap in surrogate units. First-principles grounding: Lanczos extreme-Ritz
convergence + E[H_K]=H_P concentration; Dykstra-feasibility not applicable (no ΔS claim —
this is a SENSE calibration).
**R2 — Decay-aware exit backtest ($0, minutes):** spectrum-rate exponential-mixture forecast
vs the powerlaw fit on the mod32cap tau trace; must retro-fix the ep450 miss (predicted-band:
mixture forecast within 2× of realized +0.0004 where linear λ said +0.0060). Kill: if the
mixture is no better than powerlaw, the spectrum stays a basin/saddle sensor only, not a
horizon model.
**R3 — Surrogate→d_seg calibration ($0):** regress surrogate loss vs n600 d_seg over the
run's existing verdict rows; gives the unit map every DECIDE floor needs.
**R4 — Next-run cadence:** per-stage-boundary spectrum probe (K=32, k=12) off the training
critical path, default-ON (score-neutral observability); emits the
`gn_spectrum.checkpoint_lanczos` producer rows the controller consumes live.

## 5. Interfaces

- **To vehicle/basis seat:** the in-basin condition gates any TerminalSolve-class finisher;
  subset-solve NO-GO stands (full-P in-trainer only). If R1 kills the indefiniteness at
  full-P, that is affirmative evidence the residual is capacity/basis-limited — Arm A's case
  strengthens either way.
- **To schedule/curriculum seat:** `quadratic_basin` ExitEvent as the Muon-fire predicate
  (fixed 726 demoted to CAP); rewarmup-length stability-edge rule once boundary spectra
  exist; #315 plateau+nucleus stays the in-run trigger — my signals are CONDITION INPUTS.
- **To DSL seat:** two gap-kind ExitEvent criteria + `GNSpectrumProbe` observability Lever
  factory (default-ON, score-neutral) + the launch-time activation-ledger ingest.
- **From pose seat:** pose ON flips DECIDE weights via λ_pose=5/√(10·d_pose) (crossover
  d_pose≈2.5e-4) — the arbitration is ready either way.
- **From rate seat:** λ_bytes 6.659e-7 S/byte prices every byte-costed lever in the queue.
- **Provided to all:** the probe harness + first spectrum artifacts
  (`experiments/results/t5_s3_hvp_lanczos_20260707/`), rerunnable on any checkpoint.

Pointer 0.19110 UNMOVED — this seat's output is MEANS until the stack lands a byte-closed
`upstream/evaluate.py` n600 exact row.
