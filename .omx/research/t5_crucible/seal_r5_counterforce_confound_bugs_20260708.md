# SEAL R5 — v7.5 birth-stack counter-force + ramp + P0 forces — CONFOUND+BUGS lens (ROUND 1)

**Date:** 2026-07-08 · **Axis:** `[macOS advisory]` $0 (ruff + unit tests + dry-run only; NO launch;
pid 63069 + run dirs UNTOUCHED). · **Pointer contest-CPU 0.19110 UNMOVED — this review is MEANS.**
**Delta reviewed:** `git diff 2fb876c43..HEAD -- '*.py'` (HEAD `ed0f24452`) — counter-force commits +
ramp `22e4e8827` + P0 phase-2 `3d63478fd`. This delta earns its OWN 3-clean-pass seal (the prior 1/3
was on the v7.4 micro-diff — NOT inherited).

**STORES CONSULTED:** `.omx/research/{v75_birth_counterforce,p0_forces_phase2_build,p0_forces_derivation}_20260708.md`
· `reports/delta_R_noise_floor.json` · `road_anomaly_probe` (via the equation docstring) · CLAUDE.md
non-negotiables (NO-FAKE, confound self-protection, value-provenance, never-invent-flags) ·
`docs/operating_manual_craft_handoff.md`. OUT OF SCOPE (pre-existing/tracked, NOT delta findings, per
prompt): the 123-unmapped/#332 gap; the #185 drift alarm (#344 503, #298 69).

## VERDICT: **NOT_CLEAN** (1 MAJOR + 3 MINOR). Pass 0/3 → resets. No BLOCKER; the SEALED v7.5 config
is launch-safe (the MAJOR bites only when a P0 force is *activated*, the explicit next step).

---

## Findings table

| # | sev | file:line | issue | scope |
|---|-----|-----------|-------|-------|
| MAJOR-1 | MAJOR | train_…R_mlx.py:2764 (LOSS_TERM_KEYS) vs :4805/:4837 | `margin_satisfice` + `temporal_screw` written to `terms_out` but ABSENT from `LOSS_TERM_KEYS` (`area_constraint` WAS added) | instance/delta |
| MINOR-1 | MINOR | chan_vese_…_20260708.py:92 (+ memo §1) | prose "total returned ~0.1145 ≥ 0.1189 deficit" — 0.1145 < 0.1189 (~96%, the `≥` is false) | advisory |
| MINOR-2 | MINOR | witness_autoconfig.py:2396 (`BirthCompletionEvent(...)`) | `tau_persist` left at DSL default 0.8 not passed from `_CRUCIBLE_V7_BIRTH_COMPLETION_TAU`; post_level derived from the constant → silent divergence if the constant is later edited | maintainability |
| MINOR-3 | MINOR/WATCH | train_…R_mlx.py:4802 (FORCE 2 hinge on `_signed`) | satisficing hinges on seed-composed `_signed`, not deploy `_signed_wa`; for the shipped partition `_signed_wa` is the relevant margin | FORMULATION (owed FORCE-2 A/B, not this seal) |

---

## MAJOR-1 — LOSS_TERM_KEYS drops two P0-force terms → breaks the loss-terms self-check on activation

`total_loss_fn` writes `terms_out["margin_satisfice"]` (L4805) and `terms_out["temporal_screw"]` (L4837),
but `LOSS_TERM_KEYS` (L2764) gained ONLY `area_constraint` — not these two. `_loss_terms_row` builds
`t = {k: terms.get(k,0.0) for k in LOSS_TERM_KEYS}` and `sum_terms = sum(t.values())`, and its own
docstring guarantees `sum_minus_total` "should sit at fp tolerance -- the terms ARE the total's addends."

**Failure scenario (on the explicit one-per-increment plan):** activate FORCE 2 (`--seg-margin-satisfice-weight>0`)
or FORCE 1 → `total` (L) includes `ms_term`/`ts_term` but `sum_terms` (iterating LOSS_TERM_KEYS) EXCLUDES
them → `sum_minus_total` no longer sits at fp tolerance = a **corrupted confound-immune-system invariant**
(the loss-terms reconciliation is an L1 self-protect signal), AND the per-term telemetry silently DROPS
the force (observability gap — violates the derivation's own telemetry spec + max-observability).
Untested: `test_loss_term_telemetry` checks the schema's internal consistency but nothing cross-references
the `total_loss_fn` writers, so this slipped. Precedent in-repo: `test_eikonal_stabilizer` asserts
`"eik_steik" in LOSS_TERM_KEYS` — the same guard is owed for the two new keys.

**Not a BLOCKER for the sealed v7.5** (verified: P0 forces NOT composed — argv carries none of the P0
flags; `test_none_of_the_three_forces_composed_on_in_crucible_v7`). But the delta BUILDS them for
activation, so the invariant breaks the moment the plan proceeds. **Fix:** add `"margin_satisfice"` and
`"temporal_screw"` to LOSS_TERM_KEYS (mirror the `area_constraint` add) + a test asserting every
`terms_out[...]` writer key ∈ LOSS_TERM_KEYS.

## MINOR-1 — Chan-Vese "area returned ≥ deficit" prose overclaim
lane returns 0.0805→0.00731 (+0.0732) + movable 0.0568→0.0155 (+0.0413) = **0.1145 < 0.1189** deficit
(~96%, not `≥`). Advisory only (the absolute λ scale is `ASSUMED_AWAITING_VERIFICATION`, owed to the A/B);
no code impact. Fix the `≥` wording to "≈ (slightly under, ~96%)".

## MINOR-2 — compose-site tau/post_level coupling
`_build_crucible_v7` passes `post_level=round(1.0-_CRUCIBLE_V7_BIRTH_COMPLETION_TAU,6)`=0.2 but leaves
`tau_persist` at the `BirthCompletionEvent` DSL default (0.8). Correct ONLY because both equal 0.8; a
future edit to `_CRUCIBLE_V7_BIRTH_COMPLETION_TAU` would move post_level while the completion gate stays
0.8. Fix: pass `tau_persist=_CRUCIBLE_V7_BIRTH_COMPLETION_TAU` (and prefer calling
`derive_post_level_from_persistence` over inlining `1-τ` — DRY, one source).

## MINOR-3 (WATCH, FORMULATION) — FORCE 2 hinges on `_signed` not `_signed_wa`
The satisficing hinge reads the seed-composed `_signed`; the SHIPPED (witness-alone) partition's margin is
`_signed_wa`. The seed can mask the hinge on the deploy-relevant band. The derivation specified `_signed`
(#141 shared) and pre-registered an A/B, so this is a formulation choice to settle at activation, not a
delta bug. Noted for the owed FORCE-2 A/B.

---

## What I re-derived (positive verification — the design is sound where it claims to be)

**CONFOUND lens:**
1. **Chan-Vese balance** — λ_c=F_birth/(δ·A_GT) ⇒ A*=A_GT+F_birth/λ_c=(1+δ)·A_GT (algebra checks; requires
   F_birth==birth_force proxy, declared MEASURED-ANCHOR). Dominance at ep125: lane (13.76−1)/0.25=**51.0**,
   movable (4.58−1)/0.25=**14.3**; λ_lane=683.8, λ_movable=322.6 — all match the memo. dL/dlogits ∝
   λ·relu·softmax·(1−softmax) is boundary-localized (softmax Jacobian = discrete δ(φ)). The area consumes
   the REALIZED **witness-alone** soft mass (`softmax(_slog_wa)`) = the SHIPPED partition = the correct
   quantity the birth force grows on the overlapping annulus. Default-OFF byte-identity VERIFIED (flag
   absent → `_area_lambda` None → term skipped; `_island_levers_on` extended only when area ON). In v7.5
   `_slog_wa` is non-None (wa routing live) → the area term is LIVE, not silently inert.
2. **Ramp** — `island_birth_perclass_from_signed_mx`: `f_a·term_a + f_b·term_b == combined` when the masks
   partition the weight support. CONFIRMED `any_mask == lane_mask | movable_mask` in BOTH `eased_island_masks`
   (L355-359) AND `build_island_masks` (L302-306); `_bc_capture_split` uses `movable & ~lane` (lane
   priority) ⇒ exact partition, rebuilt in lockstep with the ladder radii (both the initial build and
   `_ladder_build_iw`). Independence VERIFIED: f_a/f_b and the term denominators are θ-independent constants,
   so ramping `mult_a` scales ONLY term_a's contribution (movable untouched). `|diff|=0.0` test uses a
   realistic mean-1-over-union fixture (abs=1e-5). Combined→split switch is gated on `amp_active` (post-fire)
   with both mult=1.0 at the fire epoch ⇒ ULP-close transition (honestly scoped as ON-path, not
   byte-identity). post_level=1−τ=0.2 derivation sound. Resume: `birth_completion_apply_restore` — legacy
   sidecar (no `__bc_fired_class`) → un-fired → byte-identical PRE-FIRE; additive `__bc_*`; restores only
   watched non-fired classes; verified from the `_FnResumable` registration (write returns `{}` when
   controller None → no keys → no manifest).
3. **P0 default-OFF** — FORCE 1: `ts_w=0` → branch + providers skipped, NO extra SegNet forward →
   byte-identical; `ground_gt` ξ is a numpy-derived constant `mx.array` (fresh leaf, not in the θ graph) ⇒
   **ZERO pose-gradient coupling by construction** (no explicit stop_gradient needed); `carrier_live`
   fail-loud without a live carrier. FORCE 2: `ms_w=0` → skipped; `m_safe<δ_R` fails closed in BOTH the DSL
   `.validate()` AND the trainer param extraction. FORCE 3: trainer default `edge_weight_source=uniform` →
   `_subpix_ew_prov` None → exact pre-existing mean → byte-identical; `pa_flipmass` missing artifact →
   uniform + LOUD WARN (never a hardcoded guess); the `ref_domain` flag is deliberately un-guarded in the
   resume divergence (decode-provenance, inert for training). All three folded into
   `_nonwa_levers_on`/`_island_levers_on` so `_signed`/`_slog` ARE computed when active (no silent-skip).
   δ_R=0.019590 is a REAL measured p95 (full quantiles + provenance in `reports/delta_R_noise_floor.json`)
   — NOT a placeholder.

**BUGS lens:** ruff F clean (8 touched files). 128 delta tests + 13 launch-resolution tests green. Every
emitted crucible_v7 flag is DECLARED in argparse (never-invent — verified via compile + argv grep +
`lever_registry.completeness().unmapped` disjoint from the new flags). Warp API real + correct
signatures/shapes (`warp_frame0_native_mlx(src_hwc(H,W,3), xi(6,), geom)`, `GroundHomographyGeom.eon`,
`xi_from_pose_calibration(pose6,s_t,s_r,pitch)`). Regime coherence VERIFIED in the sealed argv:
`--logit-adjust-classes 3` == `--persistence-classes 3` (agree). FORCE 3 W_e neighbour lookup uses the
SAME `_act`/`_dir_full`/`_lst` as the subpix term (aligned to the active straddle set); minor fencepost at
the last row/col (neighbour defaults to class 0 = a defined W_e entry) — negligible. `_idet` scope safe
(referenced only under `amplify_w>0`).

**Robustness note (advisory, not a finding for the 2-class v7.5):** the per-class split + amp-ramp assume
exactly two island classes {lane, movable}; a hypothetical third island class from
`identify_island_classes` would be silently dropped from the per-class term (the split covers only
lane⊎movable). Sound for v7.5; consider a partition assertion at `_bc_capture_split` for generality.

**Pointer 0.19110 UNMOVED — MEANS. Fix MAJOR-1 (+ MINORs), re-review, then continue the 3-clean-pass.**
