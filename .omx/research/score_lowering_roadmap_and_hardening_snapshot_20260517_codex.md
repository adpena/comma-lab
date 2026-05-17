# Score-Lowering Roadmap And Hardening Snapshot

Date: 2026-05-17
Commit: f9180c7621e0655bc14c826148393f329259b356
Scope: L5 v2 / TT5L priority, non-HNeRV frontier work, lattice/graph,
Cathedral/autopilot, exact-dispatch custody, and current bug-hunt backlog.

Authority:

- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- ready_for_provider_dispatch: false
- dispatch_attempted: false
- provider_spend_attempted: false

## Executive State

Current `main` is the source of truth. The latest landed hardening commit is
`dispatch: canonicalize exact active-claim policy`, which separates the two
dispatch-claim meanings that were previously conflated:

- preclaim fan-out: an active same-lane claim is a conflict;
- provider submit: a matching active lane/job claim is required.

This closes one custody ambiguity before L5 v2 / TT5L and stack candidates
start generating paired exact CPU/CUDA anchors.

L5 v2 / TT5L remains the top score-lowering priority. The method is not
currently promoted or locked. The architecture lock packet says lock is still
forbidden because all gate evidence is not valid, the side-info effect curve is
not harvested, and probe gates are still incomplete.

## P0 Score-Lowering Roadmap

1. TT5L side-info effect curve: execute the 10-cell paired CPU/CUDA Lightning
   route after provider doctor passes. Required sequence: configure Lightning
   route, run required doctor, stage per-cell source manifests, create active
   lane claims, submit non-dry-run cells, harvest artifacts, close claims, then
   rebuild harvest cells, side-info effect curve, and architecture lock packet.
   No score claim until harvested contest-auth-eval artifacts exist for all
   required cells.

2. L5 v2 lattice probe completion: fill the missing paired exact observations
   for C1 world-model foveation, Z5 predictive-coding world model, and TT5L.
   Architecture lock stays false until probe gate, side-info curve, paired-axis
   evidence, first-anchor timing smoke, and anchor-pair custody all pass.

3. NSCS01 nullspace split renderer: run the head0 architecture disambiguator
   and smoke-before-full at the same archive bytes. The probe must measure
   frame-0 versus frame-1 PoseNet gradient norms, SegNet frame-0 perturbation
   invariance, head0 CNN-vs-MLP capacity, and paired CPU/CUDA exact follow-up.
   Full dispatch waits on smoke-green plus nontrivial frame-0 PoseNet signal.

4. NSCS02 downsampled renderer: run the paired downsample-ratio smoke and the
   eval-roundtrip-chain disambiguator before any full dispatch. The key risk is
   spatial-frequency destruction from bicubic upsample; R=2 is a hypothesis,
   not a promoted setting.

5. NSCS03 Ballé joint codec: run lambda_R plus sigma-floor co-sweep, then A100
   smoke, then byte-mutation no-op proof across entropy state, main latents,
   and hyper latents. Implement the differentiated EMA split as a documented
   substrate-specific fork before full dispatch.

6. A-STACK NSCS01+NSCS02+NSCS03: do not dispatch composition before individual
   anchors. The recommended curriculum is sequential: NSCS02 spatial budget,
   then NSCS01 gradient-routing budget, then NSCS03 entropy-coding refinement.
   Current design band is prediction-only and must be Dykstra-gated.

## P0/P1 Hardening And Bug-Hunt Backlog

1. Unify architecture-lock authority. `tt5l_campaign_readiness` and the
   architecture-lock packet should share the same strict predicate so no UI,
   autopilot, or briefing path can surface a weaker lock boolean.

2. Require full-frame output custody in L5 v2 probe/eval validation:
   inflated-output manifest, raw-output aggregate SHA, runtime tree SHA,
   command, hardware, log, and sample count should be mandatory for probe
   axes that can influence architecture lock.

3. Patch remote wrappers that only log claim checks. Z6 and Rudin remote
   scripts must verify active claims from the canonical ledger and append
   terminal claim rows on success/failure.

4. Fix Z6 smoke epoch semantics. Default smoke must be bounded to the intended
   small epoch count; full-run epoch defaults must remain explicit and cannot
   be reached accidentally through a smoke wrapper.

5. Make Cathedral/autopilot consume the shared exact-dispatch authority helper
   instead of interpreting `ready_for_exact_eval_dispatch` as authority or
   treating diagnostics as visibility-only when they should block or annotate
   unreliable posterior dispatch.

6. Split probe intake recognition from custody validity. `accepted_for_observation`
   should mean recognizable source record only; custody-valid records need a
   separate field with exact blockers attached.

7. Tighten contest-CPU provenance. CPU evidence should explicitly prove Linux
   x86_64 contest-compatible runtime, not merely "CPU and not macOS/GPU".

8. Stop stale action pressure. L5 v2 readiness and Cathedral validation queues
   should hide or supersede scaffold actions whose L1 artifacts already landed
   and display the true next blockers instead.

9. Make lattice diagnostics actionable. Compressive-sensing lattice diagnostics
   should be visible in operator briefing/preflight and should block or label
   dispatch recommendations when the lattice posterior is unreliable.

10. Continue exact-dispatch authority rollout. The shared helper now exists and
    supports active-claim policy. Remaining consumers should be migrated:
    top-k fanout, Cathedral autonomous loop, Modal auth-eval CUDA/CPU, PR106
    Lightning stack, public replay, and briefing/advisory surfaces.

## Research And Paper-Rigor Backlog

1. Keep L5 v2 Dykstra geometry as the prediction authority. Additive
   five-move composition arithmetic is retired; prediction bands must be
   polytope-intersection projections with explicit uncertainty.

2. Keep hard-earned versus cargo-culted assumptions attached to every new
   substrate, especially when borrowing literature defaults from a different
   regime.

3. Keep CPU/CUDA/macOS/proxy axes separate in every report. A paired CPU/CUDA
   cell is required before promotion wording or rank/reward language.

4. Preserve negative results as trust-region updates. TT5L's existing paired
   diagnostic anchor is non-promotional and bad on score, but it remains useful
   as a measured-config failure and timing/custody anchor.

5. Public frontier deconstruction remains active but should not crowd out L5
   and non-HNeRV class-shift work. PR95/PR101+ remain control and lessons; the
   next frontier movement needs new byte-closed candidates, not another
   narrative replay of saturated PR106-style local-basin work.

## Next Concrete Actions

1. Run the Lightning required doctor for TT5L, then stage manifests and claim
   each side-info cell only if the doctor passes.
2. Patch L5 architecture-lock predicate split and add a regression for the
   intermediate state where side-info/probe/paired plan pass but timing smoke
   or anchor-pair custody is missing.
3. Patch Z6/Rudin remote claim verification and terminal claim lifecycle.
4. Patch Z6 smoke epoch cap.
5. Patch L5 probe custody to require full-frame output manifests and raw-output
   aggregate SHA before any architecture-lock influence.
6. Migrate the next exact-eval actuator to the shared exact-dispatch authority
   helper.
7. Refresh the L5 architecture lock packet after the above patches.

This snapshot is intentionally compressed. The detailed source ledgers remain
the authoritative per-lane evidence.
