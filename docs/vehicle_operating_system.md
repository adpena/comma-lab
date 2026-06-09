# Vehicle Operating System — the permanent fix for "names outrunning implementations"

**Source:** operator binding directive 2026-06-09 (the fleet-wide-meta-bug crux session). Empirical
anchors: the 2026-06-09 full-stack audits proving HiNeRV is an L0 SKETCH (`9538bb4f3` + git `7a004e5bd`),
SNeRV is a cross-wiring defect (`83479abfe`), pact_nerv_vq is a skip-free decoder with the right objective
(`222099bc4`), and the reference-carrier comparison (`23bff0783`). The repeating failure was NOT picking
wrong ideas — it was **letting names, partial mechanisms, and bolt-ons stand in for complete vehicles**:
take a partially-implemented carrier, give it a powerful literature name, observe failure, bolt on another
idea, observe partial signal, forget whether the base was ever complete. Months disappear that way.

This document is the operating system that extincts that pattern. It is canonical and binding; every
subagent reads it (via the CLAUDE.md pointer) before touching a vehicle.

## The 5 operating-system rules (binding)

1. **No named vehicle exists until its mechanism-completeness manifest proves it.** A module import, a
   docstring, or a literature name is NOT proof. The proof is a `vehicle_fidelity_manifest.v1`
   (`tac.substrates._shared.vehicle_fidelity_manifest`) whose `verify()` passes.
2. **No vehicle may be optimized contextually until it is optimized intrinsically.** Intrinsic = "can the
   vehicle do the thing it claims, on the real inputs?" Contextual = "does it help the archive under exact
   `evaluate.py`?" (V3's job). Intrinsic precedes contextual. No long training run before L1/L2.
3. **No cross-vehicle bolt-on is allowed until the receiving vehicle's own reference contract passes.** A
   bolt-on cannot rescue an unproven vehicle identity. Compose only at L4+.
4. **No score-aware run exists unless the scorer objective is actually active.** If a run/config/memo claims
   "score-aware" or "PR95-style", the SegNet AND PoseNet objective weights must be explicit and nonzero, or
   the manifest must declare `scoreaware=false`. (Enforced by the Catalog gate
   `check_score_aware_run_has_nonzero_scorer_objective_weights`.)
5. **No row updates the score roadmap unless `authority_tier` + `metric_family` allow it.** The metric-
   laundering firewall: only a contest-axis (`contest_cpu`/`contest_cuda`) `exact_evaluate` row with full
   fields updates the score roadmap. Everything else is `mechanism_update_eligible` at most.

## The contest law these rules serve (ground truth, never edit upstream/)

`S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37,545,489`, computed by `upstream/evaluate.py` on the
inflated frames of `archive.zip` + `inflate.sh` within a 30-minute budget. SegNet scores ONLY frame1
(last frame of the pair), 5-class argmax-disagreement RATE (keys on high-frequency class boundaries).
PoseNet scores BOTH frames via RGB→YUV6 (mean=127.5, std=63.75), MSE on the first 6 of 12 pose dims. Any
architecture/objective fix must ultimately move THOSE surfaces — not PSNR (a blurry 21 dB reconstruction
still collapses SegNet argmax to d_seg≈0.5; PSNR ≠ d_seg).

## The maturity ladder (L0–L7)

Every vehicle moves through the SAME ladder. The manifest records `maturity_level`. Gates: **no long
training run before L1/L2; no cross-vehicle composition before L4.**

| L | Name | Proven when |
|---|---|---|
| **L0** | Sketch | Module exists. The NAME IS NOT A CLAIM. Allowed claim: `research_carrier_sketch` only. |
| **L1** | Mechanism-present | Required reference mechanisms are implemented AND unit-tested (each with a behavioral test, not a constant-check). |
| **L2** | Intrinsically optimized | The vehicle solves its OWN native sanity task (see below) — measured, not asserted. |
| **L3** | Archive-real | export → `inflate.sh` → frames path is bound by hashes; bytes are consumed (no-op detector); inflate is numpy-portable + scorer-free + within budget. |
| **L4** | Exact-scored | A `CandidateActionEvaluation` row exists with real d_seg, d_pose, bytes, `authority_tier`, `metric_family`. |
| **L5** | Contextually optimized | Compared under V3 against other vehicles/atoms by exact ΔS. |
| **L6** | Composable | Commutator measurements exist for its interactions with other vehicles. |
| **L7** | Promotion-ready | Paired contest-CPU + contest-CUDA `exact_evaluate` rows on the SAME archive hash. |

**Native intrinsic sanity task per family (the L2 bar):**
- HiNeRV / HNeRV: reconstruct the source video AND the LIVE render's exact d_seg/d_pose move under the
  frozen scorers (frame1 enters SegNet chambers; PoseNet sees a plausible YUV6 trajectory; pair-local
  latents are alive). PSNR alone is NOT L2 (PSNR ≠ d_seg).
- SNeRV: LF/HF/MFU/HFR/TUB are source-forward causal; one-bit payload flips change receiver frames AND
  scorer terms; TUB reifies-or-drops with a proof.
- PACT-VQ: codebook is genuinely used (healthy perplexity / assignment), codes temporally coherent, the
  decoder reconstructs AND its live d_seg moves.
- PR110++: each mode has a measured `ActionEffect`; the selector replay reproduces the known baseline; the
  coded stream pays rent.

## Intrinsic vs contextual optimization (the two phases)

**Intrinsic** (the vehicle's own L1→L2 proof): "Can it do what it claims?" Owned by the vehicle's
completion subagent. Examples are the native tasks above. A recon-fit / live-d_seg / codebook-usage /
selector-replay check — NOT a cross-vehicle comparison.

**Contextual** (L4→L5, V3's job): "Does this vehicle help THIS archive under exact `evaluate.py`?" A
`CandidateActionEvaluation` with `delta_score_total`, `pays_rent`, `base_archive_sha256`, `stale_for_base`,
`authority_tier`, `metric_family`. **No intrinsic win is automatically a contextual win** — this is the
discipline that kills the bolt-on failure mode.

## Objective-activation rules (Mistake B is permanent-forbidden)

The 2026-06-09 audits proved the shared MLX harness silently defaulted the SegNet/PoseNet distillation
weights to 0.0 — so "score-aware" runs trained recon-MSE-only (SNeRV ep22399: `observed_segnet_
distillation_weight=None`, d_seg=0.71). Fail-closed forever:

- If a run claims score-aware: SegNet/PoseNet weights MUST be explicit and nonzero, OR `scoreaware=false`.
- If a telemetry field is named `d_seg`: it MUST be true argmax-disagreement or exact `evaluate.py` d_seg.
- If it is a hinge/margin/CE surrogate: name it `*_hinge_loss` / `*_margin_loss` / `*_ce_loss` /
  `seg_axis_train_loss_proxy` — never `d_seg`.
- If a run claims PR95-style: it must list all stages and the actual per-stage loss formulas.

## The 10 non-negotiable claim rules (paste-into-repo; binding)

1. A vehicle name is not a mechanism proof.
2. A module import is not implementation completeness.
3. A docstring is not evidence.
4. A run name is not an objective contract.
5. A proxy loss is not d_seg or d_pose.
6. Structural survival is not scorer-effect survival.
7. Scorer-effect survival is not exact-eval authority.
8. A bolt-on cannot rescue an unproven vehicle identity.
9. A vehicle must pass intrinsic maturity (L2) before contextual V3 composition (L5).
10. Every candidate must pay rent under exact score.

## Fail-closed claim gates (per family)

- No **paper-HiNeRV** claim without grid-PE/interpolation/depthwise-MLP parity proven.
- No **PR95/HNeRV-faithful** claim without pair-local-latent + HNeRV-block fidelity + bilinear-skip + refine.
- No **score-aware** claim with SegNet/PoseNet weights zero.
- No **SNeRV** claim without LF/HF/MFU/HFR/TUB source-forward proof rows.
- No **VQ** claim without codebook/perplexity/assignment audit.
- No **PR110++** claim without selector replay + exact per-mode `ActionEffect` rows.

## Dashboard discipline (no hidden state)

A generated dashboard (`pact_compiler_dashboard.{json,md}`) lists, per vehicle: maturity level, current
ALLOWED claim, latest artifact, `authority_tier`, `metric_family`, current blocker, next command, owner,
pass route, fail route. Work is dashboard-driven; no stale-memory decisions.

## Subagent contract (no vibes)

Do NOT ask a subagent to "look at vehicle X." Give it a bounded, manifest-producing task. **Every subagent
ends with a manifest, tests, or a machine-readable blocker — never a prose-only memo.** A research memo is
acceptable only as the evidence base that POPULATES a manifest in the same or a paired landing.

## Canonical surfaces (where this lives in code)

- `tac.substrates._shared.vehicle_fidelity_manifest` — the `vehicle_fidelity_manifest.v1` schema + emitter
  + `verify()` (landed `143bcc1fb`). Manifests under `.omx/state/vehicle_fidelity/`.
- `tac.framework_agnostic.canonical_kernels.bilinear_skip_residual_canonical` / `terminal_hf_refine_
  canonical` — the canonical HF-residual primitive every NeRV carrier composes (no per-carrier duplication;
  landed `11b15cd02`).
- `check_score_aware_run_has_nonzero_scorer_objective_weights` (Catalog gate) — Mistake-B enforcement.
- `tac.optimization.frozen_evaluator_contract` + the V3 ingest path — contextual (L4+) authority.
- The 2026-06-09 audit memos (`.omx/research/{deep_hinerv_snerv,snerv_all_vehicles,snerv_fullstack,
  pact_nerv_vq_fullstack,reference_carrier_comparison}_*`) — the per-vehicle evidence base.

## Why this is hopeful (not discouraging)

For the first time the failure pattern has a simple explanation: different labels, same missing carrier
mechanism, same inactive objective, same lack of exact-eval closure. That is fixable. A TRUE failure would
be: complete vehicles + correct objectives + exact-eval closure + still no descent. We are not there — we
have mostly been testing incomplete vehicles. The escape hatch: make every vehicle prove what it is before
it is optimized; make every optimizer/loss/codec prove what it changes before it is trusted; make every
exact-eval row pass V3 before it influences the roadmap; make every subagent produce manifests/tests.
