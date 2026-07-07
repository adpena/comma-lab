# Deep-math converged next-run config (B / task #285) — the "Amortizing the Argmax" pass → argv

**2026-07-04. The cross-chapter-CONVERGED next-run config from the deep-math pass (§7 of
`deepmath_amortizing_argmax_paper_draft_20260704.md`), turned into verified trainer argv.**
This is a **PLAN, not a launch.** Every lever is **net-S #205-gated** (measured through the byte-close,
not asserted) and **operator-GO-gated** for any paid/heavy dispatch (CONTAINMENT). #205 (pid 29129) is
SACRED — none of this touches the live run; it targets the **next fresh run** + the in-run #270 A/B.
Pointer contest-CPU **0.19110 UNMOVED** — all MEANS.

## Flag verification (never-invent-flags — done 2026-07-04)
Every flag below was grepped-confirmed present in `experiments/train_levelset_witness_realized_through_R_mlx.py`
(Ch.4 τ/eikonal, Ch.6 transition-easing) or `experiments/train_witness_realized_through_R_mlx.py`
(Ch.5 M1 `--n-dir-freqs`/`--freq-across`). **UNBUILT (no flag yet):** Ch.5 M2 NTK-whitening + Ch.1
dash-comb — these are BUILD tasks (#286-adjacent / #287), NOT config, and are excluded from the argv
here (a gauge/argv for them would be an invented flag).

## The ranked config (each an isolated A/B vs the byte-identical baseline, for clean attribution)

Per the per-stage-treatment + optimal-form disciplines, each lever is an **isolated arm** vs the current
baseline so the measured Δ is attributable. Ordering = highest-converged-confidence + do-coupled-first.

### Tier 1 — land in the next fresh run (config-only, flags exist)

**(1) Ch.6 transition-easing [L1+L2] — attacks the MEASURED ep300 bump; do FIRST (free/built).**
The ep300 bump (d_seg 0.0056→0.020, 3.4×, DAG FEED-ft) is a *numerical-continuation* failure: the
CE→tau switch (`--tau-softplus-start-epoch 300`) and the lane-band engage (`--lane-band-start-epoch 300`)
collide at one epoch at full LR with stale momentum. Fix = deconflict + reduced-step corrector:
```
--lane-band-start-epoch 350          # deconflict from tau@300 (one homotopy param at a time)
--stage-transition-rewarmup-epochs 20
--stage-transition-rewarmup-floor 0.1
--stage-transition-rewarmup-shape cosine
```
Baseline arm: band@300, rewarmup-off (current). Equations: `ce_softmax_mirror_descent_natural_gradient_v1`
+ `muon_finisher_schedule_warmstart_and_lr_anneal_v1`. DSL: `StageTransitionEasingGauge` (this pass).

**(2) Ch.4 Γ-optimal τ-schedule + raised eikonal [COUPLED] — the phase-field lever.**
τ=ε=ħ (the Modica-Mortola interface width): the anneal shape should be geometric (equal epochs per
octave of interface width = scale-space/GNC-correct), τ_end should floor at the resolution scale (0.05
= 0.025px interface = 40× sub-grid aliasing, wasted), and the eikonal must be raised to make τ a real
interface width (it enables the τ-floor → COUPLED, one arm):
```
--tau-anneal-shape geometric
--softmax-temp-end 1.0               # floor at resolution scale (from 0.05)
--eikonal-weight 0.05               # from 0.01 — enables the τ-floor
```
Baseline arm: cosine / temp-end 0.05 / eikonal 0.01 (current defaults — the DSL BASELINE chart, pinned
by reading the trainer). Equations: `tau_eps_hbar_one_dequantization_two_scales_v1` +
`multiphase_modica_mortola_perimeter_gamma_limit_v1` + `mcf_minority_erasure_inevitability_v1`. DSL:
`GammaTauEikonalGauge` (this pass). NOTE: this changes the interface width at every epoch → it is a
*whole-run* arm, not a fine-tune; measure d_seg convergence + late-τ volatility (the code's own note is
geometric "slows late-τ d_seg volatility" — convergent).

### Tier 2 — the live in-run test (already armed, on the SAME #205 run)

**(3) Ch.6/Muon warm-start [#270] — fires at the ep726 AdamW→Muon boundary.**
`--muon-warm-start-momentum --muon-lr-final-frac 0.1` — a direct in-run A/B of the L2+L5 transition-easing
principle (warm-start kills the +0.000357 cold-start spike; cosine LR-anneal escapes the flat-LR plateau).
Already GO'd, deterministic-repro provenance in `[[muon-restart-config-change-deterministic-repro-provenance]]`.
**Watch #270's ep726 d_seg trace before committing Tier-3 builds** — it's the cheapest real signal on
whether transition-easing pays.

### Tier 3 — need a build first (NOT config; excluded from argv until built)

**(4) Ch.5 M1 along-tangent [#277 built] + M2 NTK-whitening [UNBUILT] :** `--n-dir-freqs 2→4` with
`--freq-across 8` (Nyquist cap `freq_across·2^(n_dir_freqs−1) ≤ 64` — do NOT blow the across leg).
M2 NTK/multiscale band-pass whitening (per-scale amplitude ∝ 1/√λ) is the microlocal preconditioner
(dominant SPEED lever ~3-10×) — **UNBUILT flag** (#204/#207 sig-proc lineage). Equation:
`shearlet_nterm_upper_bounds_task_rate_v1`.

**(5) Ch.1 dash-comb + AHA logit-offset — CORRECTION 2026-07-04: BOTH ALREADY BUILT (NO-FAKE fix).**
The proactive-recall sweep for the "design and build" directive found the tropical max-plus dash comb
is FULLY BUILT (`analytic_lane_render_band._line_row_params:174-180` = the periodic mod-period/duty comb;
`fit_lane_line:262` fits period/phase/duty to real GT dashes; `serialize_lane_band_rd` LBND2 COUNTS the
3 comb params per line with quantization tolerances — Wave-F #229/#234). The Laguerre logit-offset head
is BUILT too (`src/tac/boundary_math/laguerre_logit_offset.py`: `--head etf`, `--head additive-margin`,
`--logit-adjust-per-class` Menon; #218). The earlier "#287/#218 UNBUILT" was WRONG — corrected. The
deep-math Ch.1 value was CONFIRMING these are theory-optimal (the basis IS a shearlet, the comb IS the
max-plus tropical comb, the head IS the Laguerre power-diagram cure), NOT new buildable levers.
**The genuinely-UNBUILT seam = the geometry-native SOLVERS (not representations):** damped-Newton
semi-discrete OT head-offset (replaces the Menon heuristic; the asymmetry cure) · auction-MBO
volume-preserving flow (the proven-erasure cure as a solver) · Airy caustic asymmetry profile · RKMK
Lie-group ξ-transport (dash phase = on-manifold exp(ξ), near-free). Each $0-gated before wiring.

## Sequencing verdict
Next fresh run: **Tier-1 arms (1)+(2) as isolated A/Bs vs baseline**, layered on the current #205 optimal-form
config. Tier-2 (#270) is the live ep726 signal to watch NOW. Tier-3 waits on builds (#286/#287) + the #270
read. **DON'T build:** deconvolve-R, output-space F⁻¹ NG, sub-2px basis refinement (§5 disarmed).

## Gating (binding)
- **Net-S #205-gated:** each arm's verdict is the byte-closed `upstream/evaluate.py` n600 row through the
  real decode (CPU authority, MLX/MPS never a score) — NOT the training-loss or a proxy.
- **Operator GO for dispatch:** CONTAINMENT — no autonomous heavy/paid launch. The governor P0 gate binds
  any n600 launch (system memory admission).
- **Deterministic repro:** seeded + resumable + per-stage checkpoints (the launch non-negotiable).

Sisters: `deepmath_amortizing_argmax_paper_draft_20260704.md` (§7) · DAG FEED-03y/03z · tasks
#285 (this) / #286 (Ch.4 lever) / #287 (Ch.1 dash-comb) / #270 (Muon warm-start) / #277 (along-tangent).
