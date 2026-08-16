# ddm_b2e charter — burn-2 TRAIN-FOR-EDITABILITY: build-to-admission + sealed ticket (NO LAUNCH from this arm)

Operator authority: 2026-08-16 "full authorization to pursue all pattern follow ons work and regimes
level work and everything" — this is ns1 P1 (#1074), the only named supplier class for the
−15,157 B rate rung. Owner of the FIRE: MAIN (governed Metal slot). This arm BUILDS to admission
and seals; it never launches training, never runs a full n600 scorer pass.

## The regime thesis (measured, do not re-derive)

Pose brittleness on the hv1/HPAC vehicle is a TRAINED property. Measured (ns1 audit, memo
`.omx/research/ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md`, sha
91741c062c38ab88ce7e225921a0024d6dc45dacc858658e02ee83ff08b2dba0):

- All three post-hoc weight edits REFUSED with pose destroyed 3.8–5.0× (base advisory d_pose
  1.4747e-4 → 5.55/6.84/7.31e-4) — receipts in
  `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/MP2_ADVISORY_ADJUDICATION.json`
  (sha 54228227d8d18bbceeca0944371ee2dcb5e773c3135bb1214416247677aeb743).
- Sensitivity is ~94× ANISOTROPIC: `18_blocks_1_film_weight` rows are the pose-critical subspace
  (0.28% rel perturbation there ≈ 28.5% in `01_frame_embed_weight`).
- The e960 burn ran QAT on TOKENS (which is why token quantization ships) but NEVER on the
  semantic weights — that is exactly why every post-hoc weight edit dies.
- Screen (the admission arithmetic): Δd_pose budget ≈ 5.1e-9·ΔB at this operating point.

The burn-2 regime trains the weights to TOLERATE the edits, converting the refused byte pools
(q3/q4 −823 B · FiLM-row −130..−2,051 B · rank pool · width multi-KB) into harvestable levers.

## Deliverables

1. **Five default-off trainer levers** on `experiments/ddm_rx2_mc36_label_hpac.py` (sha at charter
   time 4a0db3c8fd42c4b4d38edd7a321a5cad84b26e9ee3f183ca73f931942b55f354 — grep its REAL argparse
   first; NEVER invent flags; every new lever lands as a real wired flag, byte-identical when off):
   - F1 `--weight-perturb-robustness`: per-step Gaussian/quantization-shaped noise on the
     SEMANTIC tensors during forward (STE-clean), with a per-tensor scale map that UP-WEIGHTS the
     located FiLM-critical tensors (blocks_1 FiLM family) — the eval_roundtrip principle applied
     to weight space.
   - F2 `--weight-qat-q3q4`: fake-quant STE on semantic tensors using the EXACT mz2 q3/q4 mixed
     map (selection map sha 70c1bd37d9308c7a76536dd78f97a8ae50a98e30c534c39210873604cb2ecd3e in
     the mp2 generation; builder `experiments/ddm_mp2_mixed_precision_receiver_close.py` sha
     ac2e60661393790e4ba547f0e42adacb9fae72984619d809cd01afd0619521aa) so the −823 B recode
     becomes free at export.
   - F3 `--film-row-dropout`: structured row dropout over the FiLM row families the keep-ladders
     prune, so keep87/keep75 sets become prunable.
   - F4 `--carrier-rank-penalty`: nuclear/spectral penalty on the carrier factorization (the
     22,032 B basis+coeff pool) — opens the lossy rank/refit rung the §A screen currently kills.
   - F5 (ARM-MATRIX CANDIDATE, not default): gate-aware conditioning per js8's live hypothesis
     ("the refused adapter was trained for uniform application, never the gated distribution" —
     `.omx/research/ddm_js8_implicit_edge_conditioning_20260814.md`). Include only if it composes
     cleanly; otherwise record as the named follow-on.
2. **The EDIT-REPLAY admission harness** (the measurable no-fake criterion): a tool that takes any
   burn-2 checkpoint, re-applies the EXACT mp2 edit constructions (q3/q4 map + keep87 + marginal
   set), and measures Δd_pose/Δd_seg on a seeded stratified n32 (bounded, labeled subset — the
   m96 axis law applies: pose subsets bias 2.5–4.2×, so the harness reports the bias-tagged read
   and the n600 advisory stays MAIN-fired). SUCCESS BAR pre-registered: post-burn edit pose
   damage collapses ≥50× vs the ns1 calibration on the same edits; else the regime thesis is
   INSTANCE-refuted and says so.
3. **Phase-B harness (ns1 P3, the #850 cap-lift)**: uncapped realized-acceptance pose-aimed solve
   — Gauss-Newton with a real CONVERGENCE TEST (no iteration cap; #850 measured every prior pose
   GN stopped at 2–3 relins while descending 13–23%/iter), realized acceptance (accept iff
   realized joint ΔS<0), seg-hold floor, per-pair, priced with the qs 4 B/pair coder. BUILD ONLY;
   it runs at the burn-2 endpoint. pk4's dead formulation (linear-FITTED overlays) is out of
   scope; qs5's exact-solve compensation (in-compile, never transferred — the qs4 lesson) is the
   proven mechanism to extend.
4. **Sealed launch ticket** for MAIN: warm-start from the ep0634 checkpoint (compose CANDIDATES in
   `experiments/ddm_hv1_harvest_compose.py`: epoch 634, sha 5007beae…; workspace
   `/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634`) WITH optimizer state if retained
   (the wd3 law: Adam state carries ~3× pose descent — verify what the checkpoint holds and say
   so); derived schedule (~60–120 ep, per-stage checkpoints P0, resumable-from-disk, EMA shadow
   saved); watcher configs (liveness + quality with joint/top1 regression guards per
   `tools/select_hpac_checkpoint.py`); memory preflight at the REAL config; governor admission.
   Endpoint obligations: selector argmin + edit-replay harness + export-fit-encode identity race
   (the proven hv1 chain).

## OPTIMAL FORM

- Reference form: the e960 burn itself — same trainer, same n600 real labels, same receipt schema
  (`ddm_rx2_final.v1`, base FINAL_RESULT at
  `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/FINAL_RESULT.json`). Every delta vs
  that reference is a declared MECHANISM-ADDITION (the five levers + harnesses), not a reduction.
- SCOPE reductions permitted and labeled: bounded lever smokes (n32/n60, epochs-capped) for
  wiring proof only — NO family verdict from any subset (wd3 spec law); the n600 rows are
  MAIN-fired.
- TOY-BRACKET: none. If any lever cannot be built at real form in budget, deliver it as a
  declared unbuilt follow-on with the exact blocker — never a stub that pretends.

## Constraints (binding)

Serializer commits w/ --expected-content-sha256; 2 review passes on every .py (never
REVIEW_GATE_OVERRIDE on .py); resumability P0; ALWAYS KEEP THE PAYLOAD; upstream/ read-only;
scorer slot untouched (bounded n32 harness passes only, and only if the slot is free — check
`tools/codex_arm_queue.py status`); SSD tiers per policy; no Modal spend. STORES CONSULTED must
include: ns1 memo · MP2 adjudication · mz2 verdict (5c073e915) · wd2/wd3 verdicts · js8 handoff ·
#850 row · qs5 verdict memo · the rx2 trainer + its live launch receipt. Final message persisted
per contract with NEXT_IF_RESUMED; done receipt via the keeper.
