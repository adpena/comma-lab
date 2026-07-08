# P0 FORCES — phase-2 BUILD (task #360) — the three in-trunk forces as DSL Lever factories

**Author:** P0 FORCES PHASE-2 BUILDER · **Date:** 2026-07-08 · **Axis:** all `[macOS advisory ·
NON-PROMOTABLE]` · **Pointer contest-CPU 0.19110 UNMOVED (means).**

STORES CONSULTED: `.omx/research/p0_forces_derivation_20260708.md` (THE phase-1 spec — built to
verbatim) · `SPEC_v75_optimal_single_trunk_20260708.md` §8 (OPERATING CONTRACT) + §9 (BUILT-vs-
ACTIVATED) · CLAUDE.md non-negotiables · `docs/operating_manual_craft_handoff.md` · the ramp landing
`22e4e8827` (the byte-identity + resume + DSL-factory precedent) · lever_registry / activation_ledger /
`_resume_lever_divergences` / the subpix term (trainer ~L4640) / `warp_real_luma_frame0`.

## What this is
The MECHANICAL implementation of the completed phase-1 derivation (NOT a re-derivation). The four
derived forces → **three DSL `Lever` factories** (the 4th R-phase FOLDS into #3 as a provider option),
ALL **DEFAULT-OFF**, each byte-identical on the off-path. This is a BUILD unit — the pointer does not
move here; the forces are activated ONE per crucible increment with a measured n600 A/B (§9).

## The three forces (built to spec)

### FORCE 1 — `TemporalScrewConsistency` (NEW factory)
`L_temp = w_t · mean_{annulus, c∈GROUND} ‖φ_c(f1) − Warp_ξ(φ_c(·,f0))‖²`.
- Warp = `tac.boundary_math.warp_real_luma_frame0.warp_frame0_native_mlx` (bit-checked vs the numpy
  oracle) built at SEG res; the 3 GROUND softmax-prob channels {Road,Lane,Undrivable} warped as one
  `(H,W,3)` field. Movable(3)/MyCar(4) are NON-ground → the plane homography is wrong for them → NEVER
  warped (`classes` ⊆ {0,1,2}, factory rejects a non-ground member).
- φ(f1) = softmax of the SHARED realized `_slog`; φ(f0) = softmax of a witness f0 forward (the EVEN
  index `c1−1` is the raw witness render `_render_R`, NOT the pose-carrier dispatch — so it is the
  witness's OWN f0 field, exactly what temporal seg-consistency needs).
- **ξ source DEFAULT `ground_gt`** (confound-SAFE): the per-pair GT screw from `xi_from_pose_calibration`
  (the SAME calibration the pose carrier uses; `--gfc-s-t/-s-r/-pitch`), a FIXED correct warp, stop-grad
  const ⇒ a **PURE seg regularizer with ZERO coupling to the (open) pose facet**. `carrier_live` = the
  DUAL-USE arm (live `model.pose_carrier.xi_effective(pi)`, grad → dxi — the seg face of the unified
  screw), **off-by-default**, requires a live pose carrier (fail-loud), and carries the **d_pose tripwire
  as telemetry** (`d_pose_guard`; revert at a stage boundary if d_pose rises — L68: pose is OPEN on this
  vehicle, so the dual-use arm bets the fragile pose optimum on the unification holding). The grad-to-ξ
  path is wired (`xi_effective` returns a differentiable twist); the tripwire *actuation* is an advisory
  stage-boundary controller decision (NEVER per-step), not an in-loop auto-revert.
- w_t cold-start 0.1; ramp at STAGE BOUNDARIES ONLY toward gradient-share ≈0.44. Per-term gnorm telemetry
  is the owed observability the ramp retune reads (the trainer already logs a global gnorm; per-term is a
  future observability add — noted, not blocking this build).

### FORCE 2 — `MarginBandSatisficing` (NEW factory)
`L_sat = w_s · mean_annulus relu(m_safe − m_wit)`, m_wit = the SHARED realized `_signed` GT-class margin
(#141). **m_safe = 3·δ_R ≈ 0.06** (headroom 2). **δ_R = 0.0196 MEASURED** (p95 of the uint8-at-camera
margin perturbation over the annulus; `tools/measure_delta_R_noise_floor.py` → `reports/delta_R_noise_floor.json`
— RE-RUN the tool for n600, never rebuild). Zero gradient where m_wit ≥ m_safe ⇒ the seg-gradient budget
reallocates BY CONSTRUCTION off the stable interior onto the ~2.6%-area band (UNIWARD satisficing).
MASK-BY-STAGE at l7 (does NOT replace CE — preserves the τ-anneal region formation). Fails LOUD if
m_safe < δ_R (both in the factory `.validate` AND in the trainer param extraction — the hinge would sit
inside the noise floor = pointless).

> **WATCH (SEAL R5 MINOR-3; verdict_scope: FORMULATION, not a delta bug — settled at the owed FORCE-2
> A/B, NOT this seal):** the satisficing hinge reads the seed-composed `_signed` (#141 shared margin),
> while the SHIPPED witness-alone partition's margin is `_signed_wa`. The seed can mask the hinge on the
> deploy-relevant band. The derivation pre-registered `_signed` + an A/B; the `_signed` vs `_signed_wa`
> choice is settled by that owed FORCE-2 A/B when FORCE 2 is activated (one-per-increment). No code
> change here.

### FORCE 3 — `TieLocusDisplacement` (WRAPS the built subpix term + adds W_e + ref-domain)
The subpix term (`t_wit = M_w/(M_w+M_q)` toward `t_ref`, δn = |t_wit − t_ref|) is ALREADY BUILT
(trainer ~L4640). This lever wraps `--seg-subpix-boundary-{weight,start-epoch,v-band}` and adds the
MISSING piece: **flip-density edge weighting `W_e[c_a,c_b]`** (a 5×5 symmetric matrix STAMPED from the
FEED-PA destination matrix via `--seg-subpix-edge-weight-source pa_flipmass` reading
`--seg-subpix-edge-weight-path`; falls back to `uniform` + a LOUD WARN if the artifact is absent — NEVER
a hardcoded guess). The per-straddle W_e is looked up from the class-pair (lstar[p], lstar[neighbour]) at
each active straddle, concentrating placement gradient on Road-adjacent edges (Road↔Lane = 41% of Road's
flips). `edge_weight_source=uniform` (the trainer default) → `_subpix_ew_prov` stays None → the EXACT
pre-existing subpix mean → BYTE-IDENTICAL. **FORCE 4 R-phase FOLDS in** via `--seg-subpix-ref-domain`
(`seg384` correct for the training loss — already post-R via training-through-R; `camera874_dphase`
reserved for the decode-time render-placement Consumer B, SPEC-ONLY/not-built — domain-invariant for the
training loss so IDENTICAL to seg384 for training, telemetry-stamped for the future consumer). This is
NOT a second term (per §FORCE 4 verdict); ref_domain is decode-consumer provenance, inert for the
training trajectory.

## Byte-identity + resume (the non-negotiables)
- Every force is default-OFF (weight 0.0 / source uniform / domain seg384) ⇒ the loss branch is skipped
  AND the provider stays None ⇒ NO extra SegNet forward, NO graph change ⇒ **byte-identical**.
- cfg-export: every force flag is `__cfg_seg_*`-exported in `_build_resume_state_arrays`
  (getattr+default, ZERO archive bytes — the resume sidecar is not byte-closed).
- resume-divergence: every trajectory-affecting force flag is checked in `_resume_lever_divergences`
  (a resume that silently drops/changes a force fails closed; a pre-force sidecar is tolerated — only
  keys PRESENT in the sidecar are checked; `ref_domain` is deliberately NOT guarded = decode-provenance,
  inert for the training trajectory).
- engagement re-treats the spike-guard (mirrors the subpix/lane/msal `lever_gate_on_at_epoch` precedent);
  `start_epoch ≥ l7` is the operator/spec discipline (needs a formed partition).

## Triality legs
- **DSL** (by construction): 3 `Lever` factories in `curriculum_dsl.py` → `lever_registry.completeness()`
  auto-derives all three (**0 unmapped / 0 stale** for the 14 new flags — never-invent-flags: every
  emitted flag exists in the trainer argparse); `activation_ledger.known_levers()`/`duty_to_measure()`
  hold all three (default-off is a tracked-nagged queue, not a grave — #363 passes by construction).
- **DAG**: `FEED-p0forces-build` appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations**: **COUNCIL-FLAGGED, NOT registered.** The L_temp / L_sat / L_tie·W_e laws are DESIGNED,
  not MEASURED (FORMALIZATION_PENDING — "council-flagged equations are not registered until their anchors
  land"). The derivation memo + this memo carry the law; register when the per-force A/B n600 rows land.

## Scope (§9, binding) — BUILT ≠ ACTIVATED
All three are **BUILT / registered / duty-to-measured**, and **NONE is composed ON in crucible_v7**
(verified: the P0-force factory names are absent from `_CRUCIBLE_V6_DSL_LEVERS`; the autoconfig source
does not activate any force flag). They activate **ONE per crucible increment** with a measured A/B
justification (≤15% loss-share each / ≤40% total; satisficing sequenced ≥ l7; attribution requires
isolation). Turning all three on by default would be a SPEC VIOLATION.

## Verification
- 19 new tests (`src/tac/tests/test_p0_forces_phase2_build.py`) green: factory emit/validate per force,
  GROUND-class rejection (#1), m_safe≥δ_R fail-closed (#2), edge-weight-map math (#3), trainer default-off
  source assertions, loss off-path guards, lever_registry auto-derive (0 unmapped/stale), activation
  ledger duty-to-measure, resume-divergence (flags changed force / tolerates absent keys / ignores
  ref_domain), none-composed-ON-in-v7.
- 151 neighbouring green (lever_registry / crucible_v7_config / v75_birth_ramp / witness_autoconfig /
  tail_cycles). ruff F clean on all touched files.
- pid 63069 + all run dirs UNTOUCHED (read-only sacred). NO launch (governed path only).

## Pointer
**0.19110 UNMOVED.** This is a BUILD unit — MEANS, not the END. The END is a byte-closed
`upstream/evaluate.py` n600 exact row < 0.19110 from a measured per-force A/B (owed, one per increment).
