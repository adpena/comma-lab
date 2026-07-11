# T1 PHASE-ADVECTION-CONSISTENCY — SEAL + n600 target-validity + GATE VERDICT (task #439)

**Agent:** witness training-dynamics reviewer (#439) · **Date:** 2026-07-11 · **Axis:** all numbers `[macOS-CPU advisory · NON-PROMOTABLE]`, $0, NO scorer forward (cached n600 `lstars`/`margins`/`gt_poses` = the frozen CPU-torch authority's outputs; the R/ξ ops on cached data only), MPS/MLX never a score. **Pointer 0.19108282 [contest-CPU] UNMOVED** — this is the SEAL GATE, a MEANS; only a byte-closed `upstream/evaluate.py` n600 row < 0.19108282 moves the score.

**Task:** SEAL the T1 phase-advection-consistency lever (#424, `tac.boundary_math.phase_primitives`: `t_wit`/`A_ξ`; wired into the levelset trainer) and measure whether it lowers d_seg — the gate that clears/blocks the launch-ready V9·CGauge arm (#432, carries T1 @ lever 0.4 / ep726). Deliverable: 3-clean-pass SEAL record + measured n600 A/B verdict + **PROCEED / REVISE / DEFER**.

**STORES CONSULTED:** `.omx/research/flicker_transform_geometry_term_design_20260710.md` (§4 T1 — the design + WHY, flicker-floor physics) · L85 (flicker=GT-side oracle floor 0.005318; sub-0.15 need 0.0008–0.0012 is 4.5–7× below → appearance-PHASE is the ONLY sub-floor route; existence proof 0.00086) · L86 (appearance-phase endgame BUILT, default-OFF, SEAL+A/B owed) · L70 (fused-R GPU bit-identity: scatter/gather-VJP re-poison) · L4 (term_domination 40% alarm) · L5 (spike-guard median-freeze signature) · `src/tac/witness_dsl/{curriculum_dsl.py,spec_v9_cgauge.py,schedule_readback.py}` · trainer T1 sites L4431-4485 (setup/validation), L5356-5374 (loss), L6077-6148 (θ-independent provider build), L8970-8972 + L9136-9146 (stage-boundary retreat + spike-guard re-arm).

---

## PART 1 — THE SEAL (rotating adversarial lenses; 3 consecutive clean passes)

**Method:** trace real call sites (not signatures); execute the OFF-path byte-identity, the ξ-target provider, the stability wiring, and edge cases; rebuild the numpy authority twin and check parity. All findings below are the ADVERSARIAL search output; **zero SEAL-blocking defects found** → the clean-pass counter reached 3 consecutive clean passes.

### Pass 1 — Correctness + wiring
- **The loss term (L5365-5374)** computes exactly what the memo §4 T1 claims: `pa_term = weighted_mean_annulus( (t_wit(p) − t_ref)² )` where `t_wit(p) = Mw/(Mw+Mq+eps)` on the SHARED realized through-R margin `_signed` (NO 2nd SegNet forward — reuses the incumbent forward under `_seg_levers_on`), `t_ref` = the ξ-advected GT tie of pair p-1, `weight` = annulus ∩ ground ∩ warped-ref-active. Uses the shared `phase_primitives` (`witness_tie_coordinate_mlx`, `phase_advection_weighted_mse_mlx`) — **op-for-op identical to the incumbent Force-3 `subpix` `t_wit` (L5236)**, so T1 and Force-3 read ONE tie coordinate (the dedup deliverable is real, verified by import).
- **The provider (L6077-6148)** builds `t_ref` θ-INDEPENDENTLY: per pair p≥1, `ξ_cross = cross_scored_frame_xi_interp(ξ_pair[p-1], ξ_pair[p])` (se(3) group composition, NOT a linear twist sum — verified in `phase_primitives.cross_scored_frame_xi_interp` via `tac.lie._se3_numpy`), then `advect_tie_field_numpy(GT_tie[p-1], ξ_cross, geom)` (the SAME bit-checked ground homography Force-1 uses). The cross-pair coupling lives ENTIRELY in this precomputed constant → the in-loss term is per-pair-LOCAL → fits the incumbent random-permutation `value_and_grad` with ZERO batching change. **Confirmed.**
- **DSL + argparse + spec all wired (triality-complete):** `--seg-phase-advect-{weight,start-epoch,classes,band,gap-xi,ref}` exist in argparse (L11443+); the DSL `PhaseAdvectionConsistency` factory exists (`curriculum_dsl.py:3581`) with fail-closed validation; `spec_v9_cgauge.compile_v9_cgauge_432_launch_config` composes it and the compiled argv emits `--seg-phase-advect-weight 0.4 --seg-phase-advect-start-epoch 726 --seg-phase-advect-classes 0,1,2 --seg-phase-advect-band 2.0` (MEASURED). `schedule_readback.py` carries the T1 force. **No orphaned lever.**
- **Unbuilt modes fail LOUD:** `--seg-phase-advect-ref witness_cached` and `--seg-phase-advect-gap-xi offline_homography` raise `ValueError` at setup (L4467-4485) — no silent wrong-path.

### Pass 2 — Deterministic-reproducibility / bit-identity
- **OFF-path byte-identity:** `pa_w=0` (DEFAULT) → the loss branch is skipped, providers stay `None`, the provider build block is not entered, NO extra forward. Byte-identical to pre-T1. **Confirmed by code trace.**
- **numpy-fp32 authority twin exists + parity MEASURED:** every MLX function has a numpy twin. Fresh parity run: `witness_tie_coordinate` numpy-vs-MLX max|Δ| = **0.0** (bit-exact); `phase_advection_weighted_mse` Δ = **1.5e-8** (fp32 rounding, well within authority tolerance); zero-weight → **0.0** (no-op byte-safety confirmed). `gt_tie_targets` sentinel/active/[0,1] invariants all hold. `pytest test_phase_primitives.py` = **29 passed**.
- **No new resume/checkpoint state:** the pa providers are θ-independent constants rebuilt at setup from the SAME GT cache — grep confirms they are NOT persisted into resume/EMA/deploy state. On resume they rebuild bit-identically. **Resume-safe; per-stage checkpoint unaffected.**
- **Fused-R bit-identity not re-poisoned (L70):** T1 adds only `mx.{maximum,pad,where,divide,square,multiply,sum}` — pad is a shift-pad, NOT a dup-index atomic scatter (the ONE op class that breaks GPU cross-process bit-identity, L70). The A_ξ warp runs ONLY in provider-build numpy (materialized constant), never in `value_and_grad`. → T1 does not touch the fused-R kernel path. CPU authority is always safe; the per-config n600 GPU bit-identity check remains the standing owed item (unchanged by T1).

### Pass 3 — Stability (the load-bearing axis) + runnability + edge cases
- **Stage-boundary retreat wired (the eikonal-ep110-CFL lesson):** T1 engagement at ep726 sets `_bnd_phase_advect` (L8970-8972) which OR's into `_stage_boundary_now` → LR re-warmup + AdamW moment handling fire on engagement (mirrors every sibling lever exactly). The engage gate (L9140-9146) ALSO calls `recent_losses.clear()` → the spike-guard re-arms on the engagement step (kills the L5 median-freeze-at-engagement class). **Both the stage-boundary and spike-guard treatments are present and correct.**
- **Bounded + alarm-covered:** the term is a weighted MSE of a ratio `t_wit ∈ [0,1]` → the raw term ∈ [0,1], finite by construction (denominator ≥ eps=1e-6 > 0; no NaN/Inf path). It is a NAMED reg term (`terms_out["phase_advect"]`) → the **term_domination 40% alarm (L8551/9830) covers it**; global grad-clip 0.5 + per-group clip + gnorm-hijack 100× alarm bound the gradient. `w_p=0.4` cold-start caps ≤10% of loss per the memo/argparse help.
- **Shared-gradient class, not a NEW instability:** the `1/eps` small-margin gradient of `t_wit` at the flip band is the SAME quantity the already-in-tree Force-3 `subpix` term computes (L5236, identical eps) → T1 introduces no new instability class; it is bounded by the same grad-clip that already governs subpix. **Not a NEW eikonal/l7/hosc-class risk** — T1 is a bounded soft prior on a stable quantity, gated at l7 (start≥muon-cap enforced by the spec placement gate L476-481), NOT an L∞ term in a viscosity flow (the l7 defect) and NOT a saturating tanh(β·sin) (the hosc defect).
- **Runnability at n600 (the #205-OOM lesson):** T1 adds ~1.4 GB fp32 provider cache (3 maps/pair × 600 × 384×512×4 — trivial vs the 128 GB envelope; noted in-code L6137). NO 2nd forward → no throughput/OOM regression beyond that constant. Per-pair-local → respects the `--verdict-batch` chunking + the launcher memory-preflight (no full-P batched forward). `--micro-batch-pairs>1` fails CLOSED (raises, L6078). **A real n600 run carrying T1 runs within the envelope; it is not a smoke-only pass.**
- **Edge cases executed:** pair 0 (no prior) → all-zero weight provider → term is a 0-gradient no-op (L6117-6120 + zero-weight→0.0). Post-warp active mask thresholded at 0.5 (bilinear warp of a {0,1} mask); off-frame/behind-camera → PERSIST fallback (the warp's non-gameable oracle). No divide-by-zero, no OOB index.

**One FORMULATION-EFFICACY caveat (NOT a SEAL defect — routed to Part 2):** `t_wit(p)` uses pair p's OWN dominant-straddle direction (`_pa_dir_prov`) while `t_ref` is the p-1 tie warped into p's frame. At pixels where the dominant straddle axis changed between scored frames, the comparison is between sub-pixel positions defined w.r.t. possibly-different neighbor axes. This is bounded (both ∈[0,1]) and CANNOT destabilize — it can only make the soft prior less effective. It is exactly the kind of approximation the memo discloses (train-time regularizer, gap approximation tolerated) and the A/B arbitrates. **SEAL is not blocked by it.**

**SEAL RESULT: 3 consecutive clean passes, zero SEAL-blocking defects. T1 is SEALED — it is correct, byte-identical when OFF, deterministic/resume-safe, and cannot destabilize the run.**

---

## PART 2 — n600 A/B (does the phase term lower d_seg?)

### The honest framing (why a pure $0 d_seg-delta does not exist)
T1 is a **training-time gradient prior**, not a state transform. Its effect on d_seg can ONLY be measured by TRAINING with it ON vs OFF and byte-closing both — there is no $0 "apply the term to cached states → read a d_seg delta." What IS measurable $0 on n600 authority is the term's **PREMISE**: is the ξ-advected GT tie of pair p-1 a VALID, better-than-trivial predictor of pair p's GT tie (i.e., is the "predictable flicker channel" T1 targets real, or is the term chasing noise)?

### The $0 through-R target-validity measurement (MEASURED, n600, full 599 scored transitions)
Script `scratchpad/t1_target_validity.py` (cached `lstars`/`margins`/`gt_poses`; the SAME `phase_primitives` + `xi_from_pose_calibration` the trainer uses; calibration `s_t=-0.003224707899…, s_r=0, pitch=-0.01`; band=2.0; ground={0,1,2}; residuals on the IDENTICAL annulus∩ground∩active support). For each scored pair p, compare against pair p's GT tie:

| predictor of GT_tie[p] | weighted-MSE (tie²) | RMS tie error (≈px) |
|---|---|---|
| **ξ-advected GT_tie[p-1]** (T1's target) | **0.10864** | **0.330** |
| no-warp persist GT_tie[p-1] (baseline) | 0.11659 | 0.341 |

- **ξ-advection beats no-warp on 70.6% of the 599 pairs**; mean per-pair relative reduction **5.76%**; absolute wMSE reduction **6.8%** (0.11659→0.10864).
- ξ-advected RMS tie error **0.330 px** sits inside the memo §2.1 predicted GT-side phase-jitter band **0.09–0.43 px** (independent confirmation of the transform-chain-phase-noise law on the SCORED sequence).

### Reading (MEASURED, labeled)
- **[MEASURED, POSITIVE] The term's premise is VALID, not noise-chasing:** the ξ-transport is a REAL, better-than-trivial predictor of the next-scored-frame boundary sub-pixel position (−6.8% residual vs no-warp, wins 71% of pairs, in-band RMS). T1 has a well-formed target with genuine signal; it is not fitting the aleatoric blink-back residue.
- **[MEASURED, HONEST-LIMIT] The ξ-predictable channel is MODEST:** it explains only ~6–7% of the cross-frame tie variance; even the perfect advected target carries 0.330 px RMS residual against the true next tie. So the deterministic flicker channel T1 can fit is a SMALL fraction of the cross-frame tie noise (consistent with the memo's own honesty; the blink-back 42% is largely aleatoric).
- **[NOT PROVABLE $0] Whether that modest tie-placement gain crosses SegNet argmax thresholds to LOWER d_seg is nonlinear (goes through argmax) and genuinely requires a training A/B.** The tie-residual→d_seg map can amplify (if placement gains move flip-band pixels across the threshold) or vanish (if sub-threshold). This is exactly why the memo pre-registered a training A/B as the arbiter.

### The training A/B (operator-GO — PREPARED, not fired)
The definitive efficacy verdict needs a matched **phase-OFF control** alongside the phase-ON arm so T1's contribution is ATTRIBUTABLE (the V9·CGauge arm alone confounds T1 with the rest of the #432 cascade). Exact config:
- **ON arm** = `compile_v9_cgauge_432_launch_config()` (carries `--seg-phase-advect-weight 0.4 --seg-phase-advect-start-epoch 726`, verified).
- **OFF control** = the identical config with `--seg-phase-advect-weight 0.0` (byte-identical to ON until ep726; only the T1 gradient differs after engagement) — a clean single-lever A/B.
- **Pre-registered acceptance (memo §4 T1):** vs Forces-1+3 baseline at matched epoch — verdict d_seg drops AND the witness's own scored-sequence spike rate rises toward the GT 0.0053 **in the correlated direction** (`blink_fit_frac` ↑ — it must fit the GT's spikes, not add its own) AND d_pose non-rising. KILL scope = FORMULATION.
- **$0, heavy-GPU, operator-GO.** Do NOT fire without GO; do not disturb the live run.

---

## GATE VERDICT: **PROCEED** — T1 SEALED; the V9·CGauge arm is CLEARED to fire

**Reason:** (1) **SEAL passed** — 3 consecutive clean passes, zero blocking defects; T1 is correct, byte-identical-when-OFF, deterministic/resume-safe, fused-R-safe, and **cannot destabilize the run** (bounded soft prior on a stable quantity, l7-gated with stage-boundary retreat + spike-guard re-arm, grad-clip + domination-alarm covered, shares Force-3's already-in-tree gradient class). The single failure the SEAL exists to prevent — a destabilizing loss term shipped into a multi-day pointer run — is CLEARED. (2) **The $0 n600 premise-validation is POSITIVE** — the ξ-advection target is MEASURED valid and better-than-trivial (not noise-chasing), so firing the arm is a scientifically-grounded bet, not a blind one.

**PROCEED is correct (not REVISE, not DEFER):** no fixable defect was found (REVISE N/A); the term is safe and its premise is validated on n600 authority, so there is no blocker (DEFER N/A). Blocking a sealed, safe, premise-validated term would be means-hoarding.

**Honesty on "helps":** the SEAL clears the STABILITY gate and the $0 measurement clears the PREMISE gate, but the d_seg-EFFICACY claim ("does T1 lower the score") is UNPROVEN and unprovable $0 — it is delivered BY the run via the pre-registered T1 telemetry (`blink_fit_frac` ↑ / correlated spike direction / d_pose non-rising), and cleanly attributable only WITH the matched phase-OFF control. The $0 evidence is SUPPORTIVE-BUT-INSUFFICIENT; the run is the arbiter.

### Remaining owed-before/at-fire items (all operator-GO; none block the SEAL)
1. **Fire the arm with a matched phase-OFF control** (config above) for clean single-lever T1 attribution — else the arm's pointer move confounds T1 with the rest of #432.
2. **Watch the pre-registered T1 telemetry** in-run (the `seg_phase_advect` provider row already emits active-px counts; `blink_fit_frac` / gnorm-ratio-vs-subpix are the acceptance signals) — confirm the term engages at ep726 with the spike-guard re-armed and stays <40% of loss.
3. **Per-config n600 GPU bit-identity check** (standing owed item, unchanged by T1; CPU authority always safe) if the arm runs GPU-with-fused-R.
4. The `--seg-phase-advect-ref witness_cached` and `offline_homography` gap modes remain BUILD-OWED (fail loud today) — not needed for the ON arm.

---

## Triality
- **DAG:** FEED-439 (this seal + measured target-validity + PROCEED).
- **DSL:** N/A — no lever changed (`PhaseAdvectionConsistency` already exists + is composed by `spec_v9_cgauge`; the SEAL verified it, did not modify it).
- **equations:** N/A-with-rationale — the target-validity measurement is a single-instance MEANS check (n600 authority, but one video/scorer, and it validates a PREMISE not a score law); it does not clear the ≥5-run anchor bar and produces no new law. It is directional confirmation of the existing `transform_chain_phase_noise_partition_v1` candidate (GT-side 0.09–0.43 px band → measured 0.330 px RMS), not a new anchor. If the training A/B lands a byte-closed d_seg delta, THAT is the registrable law.

## Honesty block
- $0, CPU-only, NO SegNet/PoseNet forward anywhere (cached `lstars`/`margins`/`gt_poses` are the frozen CPU-torch authority's outputs; the ξ/warp ops run on cached data). No training, no dispatch, no process disturbed. All fresh numbers `[macOS-CPU advisory · NON-PROMOTABLE]`.
- **Pointer 0.19108282 [contest-CPU] UNMOVED.** The SEAL + the premise-validation are MEANS; the pointer moves only when the arm fires → byte-closes → `upstream/evaluate.py` returns an n600 row < 0.19108282.
