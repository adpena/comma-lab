---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary, DE-DERIVATION-318]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "the `8` in sqrt(eta*lambda_eik/8) is DERIVED, not MEASURED — same class as the 38 that
      already failed. And at the launch constants the adaptive law degrades to a CONSTANT-0.3-FLOOR
      (|c_a|~10 keeps eps clamped): the 'adaptive' rarely fires. Ship the FORM + the byte-identity
      proof; the net n600 cure is owed to main's probe, NOT claimed here."
council_assumption_adversary_verdict:
  - assumption: "the interior-mean |c_a| is the sharpness quantity the CFL edge cares about"
    classification: CARGO-CULTED
    rationale: "interior-mean |(|grad m|-1)/|grad m|| is DOMINATED by flat-interior pixels (|grad m|->0
      => the ratio blows up), NOT the boundary-annulus sharpness that drives the ill-posed a<1 mode.
      The band-restricted variant (--eikonal-visco-ca-band) is the physically-sharper proxy; band=0 is
      the §7.4 literal launch form and its ACTUAL effect here is 'floor eps at 0.3 + explosion insurance'
      (which alone removes the eps->0 HALF of the DE §3.1 re-entry). Honest, surfaced, not hidden."
council_decisions_recorded:
  - "op-routable #1: BUILT + tested (21 tests, byte-identical OFF, numpy<->MLX parity) the adaptive-eps
     CFL-edge tracker on the levelset trainer; DSL EikonalViscoStabGauge.ADAPTIVE_EPS; equation
     adaptive_eps_cfl_edge_tracking_v1 registered (byte-identity+parity anchor VERIFIED, n600 owed)."
  - "op-routable #2: the CLEAN v6 A/B config (§3) = v5 argv + adaptive-eps + flat lambda_eik, resume
     from the SAME bd_calib ep100 (one variable = adaptive-eps). Main launches at n600 under authority."
  - "op-routable #3: the pre-registered probe gate (§4) — main watches to ~ep150; PROMOTE iff no
     re-entry + SC1' skip<10% + d_seg descends past ep110 + no #315 per-class stall; ABORT if re-entry."
related_deliberation_ids: [witness_config_differential_equations_derivation_20260705,
                           council_v6_eikonal_cure_symposium_20260705, eikonal_stabilizer_build_20260705]
---

# ADAPTIVE-eps MECHANISM-CURE (#320) — the DERIVED, byte-identity-proven eikonal re-entry cure

**Axis discipline: everything here is MEANS (a 0-archive-byte train-time viscosity schedule). The
byte-identity-OFF + numpy<->MLX parity are MEASURED (advisory, gt_n6 + 21-test suite). The net n600
d_seg cure is OWED to main's bounded probe. The constant `8` is FORMALIZATION_PENDING. Pointer
contest-CPU 0.19110 UNMOVED.**

## 1. The mechanism (DE #318 §3.1, restated as the build target)

The eikonal `(|grad m|-1)^2` penalty flow is provably ill-posed (StEik anti-diffusive). ViscoReg's
viscous residual is discretely stable iff the CFL group holds:

    pi_eik = eta * lambda_eik * |c_a|^2 / (8 * eps^2) <= 1    <=>    eps >= |c_a| * sqrt(eta*lambda_eik/8)
    c_a = (|grad m| - 1) / |grad m|      (m = decision margin top1-top2, the witness's own field)

The v5 ep110 death is a **TWO-SIDED squeeze**: eps ANNEALS DOWN (`--eikonal-viscosity-anneal 1000`,
0.3->0.27->0) toward the lower edge, WHILE progressive sharpening (hosc-beta 1.0->5.134, tau descent)
GROWS `|c_a(t)|` and thus RAISES the edge. The margin `eps - eps_lower(t)` is squeezed to zero at
eps~=0.27 (v5 ep109). The **fix** (DE #2, the mechanism-targeting one): make eps TRACK the rising edge
instead of annealing into it:

    eps(t) = clamp( |c_a(t)| * sqrt(eta(t) * lambda_eik(t) / 8) * (1 + margin_factor), eps_floor, eps_upper )

This holds `pi_eik <= 1` by construction with the LEAST isotropic over-damping (best d_seg-drift).

## 2. What was BUILT (deliverables 1-2)

**The |c_a| proxy + parity (deliverable 1).**
- numpy reference `src/tac/boundary_math/eikonal_sharpness_proxy_reference.py`:
  `sharpness_proxy_c_a(m, band)` + `adaptive_visco_eps(...)`. Self-test PASSES; its central-diff
  interior stencil is byte-parity-checked against #318's `eikonal_normal_curvature_reference._central_grad`
  (max abs diff 0.0). Physics verified: unit-gradient plane -> |c_a|=0; m=3x -> 2/3; m=0.5x (the
  ill-posed a<1) -> 1.
- MLX trainer helpers (`experiments/train_levelset_witness_realized_through_R_mlx.py`):
  `_adaptive_visco_eps` (the eps law, pure), `_ca_from_margin_mlx` (|c_a| on a margin field, SAME
  central stencil + 1e-8 gmag floor as `_eikonal_margin_interior_mlx`), `_measure_ca_mlx` (no-grad,
  ONE `model.sdf` forward per pair over a FIXED strided subset — **witness-only, ZERO SegNet cost**;
  |c_a| is the witness's own decision margin, not a scorer forward).
- **NO-FAKE**: |c_a| is MEASURED from the real field (gt_n6 smoke: `visco_c_a` = 11.16->10.08 over
  ep1-4, a real decreasing number as the field sharpens toward the SDF — NOT a constant).

**The flag + telemetry (deliverable 2).** `--eikonal-viscosity-adaptive` (default OFF = the existing
linear-anneal path, byte-identical) + `--eikonal-visco-eps-floor` (0.3) / `--eikonal-visco-eps-upper`
(0.7) / `--eikonal-visco-margin-factor` (0.5) / `--eikonal-visco-ca-pairs` (16) / `--eikonal-visco-ca-band`
(0.0). When ON: the linear anneal is SKIPPED; eps(t) is computed per-epoch AFTER the LR is set (so
eta(t)=`opt.learning_rate` and lambda_eik(t)=`eik_w_ep` are the CURRENT epoch's values) and mutated
into the `_eik_stab` closure cell. Telemetry: an `eik_stabilizer_adaptive` per-epoch JSON row (eps,
|c_a|, eta, lambda_eik) + top-level `visco_eps`/`visco_c_a` fields on every `loss_terms` row (NOT
inside `terms`, so `sum_minus_total` stays a clean loss-addend check).

**Byte-identity at OFF — PROVEN (deliverable 2, the NO-FAKE gate).**
- STRUCTURAL (airtight): the ONLY OFF-path changes are (a) the anneal gate `if ... and not
  _eik_stab["visco_adaptive"]` — when the flag is absent `visco_adaptive=False` so `not False`=True =
  the identical condition; (b) `_loss_terms_row` gains `visco_eps`/`visco_c_a` kwargs defaulting None
  => no new row fields; (c) `_eik_stab` gains keys `total_loss_fn` never reads; (d) argparse flags.
  NONE touch `total_loss_fn`, the model, the optimizer, or the gradient.
- EMPIRICAL (gt_n6, `--mlx-device gpu`, 4ep): at ep0 both the anneal AND adaptive paths use eps=0.3,
  and the ep1 `loss_terms` are **BIT-IDENTICAL** — total `363.537496`, seg `319.979141`, pose
  `43.521303`, gnorm `5697.5688` in BOTH runs. This proves the |c_a| no-grad measurement adds ZERO
  perturbation to the training state. The OFF run's `loss_terms` rows carry NO `visco_eps`/`visco_c_a`
  keys (schema-identical to pre-change). (CPU bitwise cross-process is blocked by a GPU-only R/stem
  metal kernel unrelated to this change; MLX-GPU is documented non-bit-identical cross-process, so the
  in-process ep0-same-eps identity is the correct instrument.)
- Tests: `src/tac/tests/test_adaptive_visco_eps.py` (21, all pass): eps-law closed-form
  (edge/floor/upper/inverted/negative-sqrt clamps) + numpy<->MLX parity; |c_a| analytic + random-field
  byte-parity + band + empty-band; `_measure_ca_mlx` stub determinism; loss_terms OFF/ON schema;
  argparse defaults; the anneal-gate + current-eta/lambda source assertions; the numpy self-test.

## 3. The exact CLEAN v6 A/B config (deliverable 3) — flag-delta vs the v5 argv

v5 argv = `experiments/results/levelset_n600_witness_20260705T155150Z/launch.sh`. The A/B isolates
ONE variable (adaptive-eps) by resuming from the SAME clean ep100 snapshot v5 used (NOT the deadlocked
gold). Delta:

```
ADD     --eikonal-viscosity-adaptive
ADD     --eikonal-visco-eps-floor 0.3
ADD     --eikonal-visco-eps-upper 0.7
ADD     --eikonal-visco-margin-factor 0.5           # (default; explicit for the record)
CHANGE  --eikonal-weight-end 0.1  ->  --eikonal-weight-end 0.05   # DE #1 flat lambda_eik (fold in,
                                                                  #   ~zero-risk post-tau insurance)
KEEP    --resume-from experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz  # SAME as v5
KEEP    --eikonal-viscosity 0.3   # the visco term must be active; adaptive REPLACES its linear anneal
KEEP    everything else in the v5 argv EXACTLY (lr 1e-3/1e-4 · --eikonal-weight 0.05 ·
        --eikonal-viscosity-anneal 1000 [inert under adaptive] · tau@400 · accum-pairs 8 ·
        --resume-allow-lever-drift --resume-clear-spike-guard · --cache-gt-skeleton --fused-r-kernel ·
        all levers/curriculum as v5)
BOUND   run to ~ep150 (bounded probe past the ep110 re-entry) · --eval-every 25 (or 10 through ep100-150)
        · --stage-checkpoints
```

**Honest behavior at these constants (surfaced, not hidden).** At eta~1e-3/lambda~0.05 the CFL edge is
only `0.00375*|c_a|`; the interior-mean |c_a| measured on gt_n6 is ~10 (flat-interior dominated), so
eps clamps at the **floor 0.3** unless |c_a| exceeds ~80. So the launch config's adaptive-eps
effectively = **"hold eps=0.3, never anneal to 0, rise only if sharpness explodes."** This directly
removes the eps->0 HALF of the DE §3.1 two-sided squeeze (DE §4 Arm-2 "Simple: floor it") and insures
against the |c_a|-up half. If the floor-only behavior is insufficient at n600, the escalation is
`--eikonal-visco-ca-band 0.5` (boundary-annulus-sensitive |c_a|) and/or lowering the floor toward the
measured edge — both are single-flag changes, pre-built.

## 4. The pre-registered probe gate (deliverable 4)

Main watches the bounded probe (to ~ep150). **PROMOTE to the full n600 run iff ALL hold through the
ep100/125 -> ep150 window:**
1. **No eikonal re-entry** — the per-epoch `eikonal` loss-term stays bounded (descends from the resume
   level; NO monotone runaway; MAX <= ~5x its post-resume trough) — the litsweep canary.
2. **SC1' healthy** — skip-rate < 10%/epoch every epoch (no absorbing-deadlock state).
3. **d_seg descends past the ep110 point** — the async CPU verdict shows d_seg descending through the
   epoch where v5 re-entered (a real `[macOS-CPU advisory]` number, non-increasing).
4. **No #315 per-class binding-term stall** — the per-class classifier does NOT flag a lane/movable
   class stalling (the failure v5's aggregate d_seg hid).

**ABORT** (preserve checkpoint) if ANY of {eikonal re-enters · SC1' deadlock · d_seg regresses past the
ep110 point · #315 per-class stall}. On abort the next lever is `--eikonal-visco-ca-band` / lower floor
(§3), then the normalized-nᵀHn insurance operator (already built, OFF the path).

## 5. Triality + artifacts (deliverable 5)
- **DAG** = FEED-06c (appended).
- **DSL** = `tac.witness_dsl.gauge.EikonalViscoStabGauge` {LINEAR_ANNEAL (byte-identical default),
  ADAPTIVE_EPS} + `eikonal_visco_stab_trainer_flags(...)` (never-invent-flags verified: every emitted
  flag is a real levelset-trainer flag). The full EIK-STAB gauge pass (steik/steik-normalized/visco
  base) remains a NOTED follow-up (per #317 triality-drift).
- **equations** = `adaptive_eps_cfl_edge_tracking_v1` (registered; byte-identity+parity anchor
  VERIFIED, n600 cure anchor ASSUMED_AWAITING_VERIFICATION; the `8` constant FORMALIZATION_PENDING).
- Build: `src/tac/boundary_math/eikonal_sharpness_proxy_reference.py`,
  `experiments/train_levelset_witness_realized_through_R_mlx.py` (helpers + flags + wiring),
  `src/tac/tests/test_adaptive_visco_eps.py`,
  `src/tac/canonical_equations/adaptive_eps_cfl_edge_tracking_20260705.py`.
