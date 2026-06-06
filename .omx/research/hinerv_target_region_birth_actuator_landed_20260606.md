# HiNeRV worst-target-region hard class-birth actuator — LANDED (L1)

UTC: 2026-06-06T17:15:00Z
Agent: claude (Opus 4.8, main session)
Lane: `lane_hinerv_target_region_score_debt_smoke_20260606` (L0 → L1, `impl_complete`)
Backlog: `hinerv_target_region_birth_actuator` (priority 2 of the operator-approved
ranked burndown at `/Volumes/VertigoDataTier/pact/incoming/pact_nerv_source_bound_burndown_20260606/`;
dependency `shared_parseback_archive_selection` landed earlier at `143b1b11a`).
Claim directive: `.omx/research/claude_directive_20260606T164500Z_hinerv_target_region_birth_lane_claim.md`.

## What landed

1. `src/tac/substrates/hi_nerv/target_region_birth.py` — torch/MLX-free module:
   - `find_target_region_debts` — prices every 4-connected component of every
     target class in EXACT contest score units
     (`100 * unsolved_pixels / total_scored_pixels`, batch normalizer recorded
     per row so smoke-scale rows cannot masquerade as full-video units).
   - `select_worst_target_region(_with_mask)` — deterministic worst-debt
     selection (debt desc, then batch/class/region tie-break) + mask
     reconstruction with drift guard.
   - `region_margin_stats` — PR95 `impostor - class` raw frontier margin
     (min / p50 / mean) + region hard ratio.
   - `allowed_birth_update_name` — scoped-update predicate
     (`latents_fine` / `feature_grids.*` / `fine_injector.*` / `head_rgb_1.*`,
     mirroring the live-SegNet-scoped bootstrap list).
   - `build_target_region_birth_receipt` — receipt with crux-trace-compatible
     receiver-surface keys (`receiver_surface_uint8_changed_pixels`,
     `receiver_surface_argmax_flipped_pixels`,
     `receiver_surface_worst_region_margin_p50_delta`,
     `receiver_surface_float_rgb_delta_linf`) + false-authority fields;
     REFUSES receipts whose updated-parameter names escape the birth scope.

2. `fit_target_region_birth_from_segnet(...)` on `HinervSubstrateMLX`
   (additive method; no existing method edited):
   -

 selects the single worst-debt connected region (stop-grad numpy),
   - region-masked loss: tau-softplus² frontier crossing + prob-floor relu² +
     frontier-seed term, weighted by stop-grad region debt,
   - gradient path: params → rgb_1 → `_receiver_uint8_roundtrip_ste_nhwc01`
     (uint8 receiver STE) → live SegNet logits,
   - scoped update application only (out-of-scope tensors bit-frozen),
   - per-step admission: receiver uint8 motion INSIDE the region required
     (receiver-quantum lr growth ×2 up to 20), pose-output trust cap when a
     pose teacher (`teacher_pose_for_yuv6_pair_nhwc`) is supplied (lr
     backtracking), progress = region hard-won pixels up OR region margin-MEAN
     down (mean registers single-pixel receiver motion; the median is pure
     telemetry — a single flipped pixel cannot move it, which re-opens the
     growth/backtrack ping-pong the rule exists to terminate),
   - honest rejection attribution (`subquantum` vs `pose_guard` vs
     `no_progress`), consecutive-rejection early stop (3),
   - full restore to initial state when nothing is admitted (fail-closed),
   - payload + receipt with grad/update norms by group, updated-name proof,
     `runtime_sidecar_bytes=0` (ordinary archive-charged tensors only).

3. `src/tac/substrates/hi_nerv/tests/test_target_region_birth.py` — 13 tests,
   NO-FAKE discipline (behavior, not constants):
   - region pricing / determinism / 4-vs-8-connectivity / mask drift guard /
     margin stats / receipt scope refusal / crux-trace keys (numpy);
   - MLX: behavioral teacher → birth lifts frontier (before ratio 0.0,
     accepted ≥ 1, region uint8 motion > 0, scoped-only updates, frozen
     tensors bit-identical);
   - MLX: subquantum teacher (1e-12-scaled logits) → every step rejected,
     full bit-identical restore (design doc Test 1);
   - MLX: adversarial pose teacher + impossible cap → nothing admitted, pose
     telemetry present;
   - MLX: no-debt labels → `enabled=False`, `no_unsolved_target_region`.

Verification: 13/13 new tests; full `src/tac/substrates/hi_nerv/tests/` =
184/184; ruff clean on all three files; suite re-run 5× for flake check
(5/5 green).

## DAG blockers this attacks

`hinerv.localized_target_region_projection_actuator` +
`shared.distortion_birth_before_rate_pressure` rows
(`worst_region_debt_reduction_missing`, `target_region_min_ratio_lift_missing`,
`accepted_update_receiver_uint8_movement_missing`) in
`tac.analysis.nerv_witness_readiness_dag`. The smoke runner can now call the
actuator on real-video targets to produce the
`hinerv_short_receiver_surface_smoke` evidence (next step, not claimed here).

## Authority

$0 local MLX/CPU. `[macOS-MLX research-signal]` / planning-control only.
`score_claim=false`, `promotion_eligible=false` on every payload/receipt
(PROXY_FALSE_AUTHORITY_FIELDS). No paid dispatch. No archive bytes changed.

## 6-hook wire-in declaration (Catalog #125)

1. Sensitivity-map: N/A-with-rationale — actuator emits per-group grad/update
   norms in receipts; sensitivity-map contribution lands with the smoke run
   on real targets, not the synthetic-unit landing.
2. Pareto constraint: ACTIVE — admission is the exact nonlinear contest
   objective surface (region debt in exact score units; pose trust cap).
3. Bit-allocator hook: N/A — zero bytes moved (`runtime_sidecar_bytes=0`);
   the actuator mutates already-charged tensors only.
4. Cathedral/autopilot consumer: ACTIVE (indirect) — receipts emit
   crux-trace-compatible receiver-surface keys consumed by
   `tools/trace_nerv_crux.py` rows and the witness-readiness DAG evidence
   contract.
5. Continual-learning posterior: N/A at L1 — no empirical contest anchor yet;
   posterior row lands with the first real-video smoke.
6. Probe-disambiguator: ACTIVE — subquantum vs pose-guard vs no-progress
   rejection attribution in the receipt IS the disambiguator between "loss
   moved" and "receiver moved".

## Sister coherence

Codex (live in the same tree this session) owns: pair-local distortion servo
admission kernel (`src/tac/analysis/nerv_pair_local_distortion_servo.py`,
untracked), crux-trace receiver-surface contract, `score_geometry`
receiver-equivalence floor audit, SNeRV TUB source-forward burndown (v61).
This landing touches NONE of those files; receipts intentionally emit the
alias keys codex's consumer ingests. No imports from sister-uncommitted
modules.
