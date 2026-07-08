# D15 — MICRO-BATCH ROUTING of --logit-adjust-loss-tau + --seg-form-unify-tau  (2026-07-08)  [no-triality]

Operator: *"build all unbuilt items."* D15 (v7 compute-audit deferral): the `--micro-batch-pairs`
2–4× step lever (#313 twin) fail-closed against v7's ACTIVE `--logit-adjust-loss-tau 1.0` (and would
have SILENTLY CE-d v7's ACTIVE `--seg-form-unify-tau`). This routes both into the batched twin so
micro-batch becomes ELIGIBLE for v7/v7.1 — the trajectory A/B (not this build) remains the inclusion
evidence.

**STORES CONSULTED:** `.omx/research/v7_compute_exploitation_audit_20260708.md` (lever #4 + the seal
to-fix); the twin `tac.boundary_math.levelset_micro_batch_loss` (#313); the serial split at
`train_levelset_witness_realized_through_R_mlx.py` (`_LogitAdjustSegAdapter` @922, `make_loss_fn`'s
`unify_tau` branch, `total_loss_fn` levers `_slog = adapter.segnet` @4315/@4332 RAW); `ln`/crucible_v7
DSL levers (`witness_autoconfig.py`, `seg_form_unify_tau` active); `memory_waterfill_config.assess_
micro_batch` + `witness_memory_preflight` (#294/#205); `curriculum_dsl.LogitAdjust/MicroBatch`;
`test_batched_seed_cograd.py` (#293); MEMORY L45/L59/L70.

**MEANS, not ends.** Nothing here moves 0.19110. Only a byte-closed n600 exact row from
`upstream/evaluate.py` (contest-CPU/CUDA, NEVER MPS) does.

## The serial split this mirrors (the load-bearing fact)
With logit-adjust ON, the serial trainer scores TWO SegNet forwards on the SAME frame:
- **base seg-form** (`make_loss_fn`, given the WRAPPED `_LogitAdjustSegAdapter`) reads ADJUSTED logits
  `φ + o`, `o_c = τ·log(prior_c)`; its focal reweight reads the same adjusted logits.
- **surgical seg levers** (lane/msal/thin/mfh) + the **witness-alone** island forward read the RAW
  `adapter.segnet` (unadjusted); **pose** is class-free (adapter pass-through, unadjusted).

The twin previously used ONE raw forward for both → adjusting it would corrupt the levers, keeping it
raw would drop the base adjustment; either is a silent-wrong. #313's `_validate_logit_adjust_compat`
correctly fail-closed. **D15 routes the split into the twin:** `seg_logits_b` (RAW → levers + wa),
`seg_logits_base_b = seg_logits_b + offset` (ADJUSTED → base form + focal), passed as `sl_base`.

## Equivalence derivations (the NO-FAKE bar)
1. **logit-adjust = BIT-EXACT per pair (not merely fp-tol).** `o` is a per-class constant broadcast
   over `(K,H,W,5)`; the add is elementwise, batch- and pixel-independent, so
   `(segnet(f1_b)+o)[k] == segnet(f1_b[k:k+1])[0] + o` bit-for-bit (SegNet eval-mode BN/conv is
   batch-independent; the add introduces NO reduction re-order). Hence `sl_base[k]` equals the serial
   per-pair `_LogitAdjustSegAdapter.segnet(f1_k)` exactly. Tolerance for the routing itself: 0 (the
   only fp reorder in the whole twin is the pre-existing mean-over-B, unchanged by D15). Offset `None`
   ⇒ `seg_logits_base_b IS seg_logits_b`, `_signed_base IS _signed` ⇒ byte-identical to pre-D15.
2. **unify-τ = same fp-tol contract as the other seg forms.** `L_τ = τ·logsumexp(φ/τ) − φ_y`; the
   `logsumexp` is over the CLASS axis (last, size 5), PER-PIXEL, batch-independent ⇒ per-pair
   `lt[k]` is bit-exact vs serial; only the existing mean-over-B is a reorder. τ is read LIVE from the
   by-ref `unify_tau_state` (the eik_stab idiom), so the render-coupled per-epoch τ tracks exactly as
   `make_loss_fn`'s `tau_override`. NO-FAKE: `seg_form=="unify_tau"` without the wired callable RAISES
   (never silent-CE). At τ=1, `L_τ ≡ CE` (documented in `_seg_unify_tau_perpixel`).

**Tests (16 new + updates):** logit-adjust {ce,tau_softplus,l7_softplus,margin_hinge}×{2,3} batched≡
mean-of-pairs; logit-adjust single≡canonical WRAPPED-adapter make_loss_fn {ce,tau,margin_hinge};
offset-None byte-identity; no-op-moves-loss; LADDER composition (logit-adjust + amplify/persist wa);
unify-τ {2,3} batched≡pairs; unify-τ single≡canonical @τ∈{1.0,0.5,0.3}; unify-τ fallback-to-lc.tau_use;
unify-τ missing-callable-refuses; validator-now-noop; waterfill-still-pins-B=1. `test_levelset_micro_
batch_loss.py` 70 pass; `test_feed07b`/`test_v7_compute`/`test_seg_form_unify_tau`/`test_crucible_v7`/
`test_memory_waterfill` all green. Seed co-grad #293 (offset None) re-verified 2× green (the one flake
was pre-existing per-group variance: seed_rel range 2.85e-5..3.40e-4 vs a 2e-4 threshold; byte-identity
of the offset-None path is structural, not mine).

## Fail-close list vs v7 argv (narrowed to genuinely-unrouted)
| lever | v7-active? | twin status after D15 |
|---|---|---|
| `--logit-adjust-loss-tau 1.0` | YES | **ROUTED** (offset on base form; bit-exact) |
| `--seg-form-unify-tau` | YES | **ROUTED** (`unify_tau` branch; live τ) |
| focal / bd / eik-stab / wa-island | (v6/v7 gated) | routed #313 (verified still true) |
| `--margin-saliency-reachability` | no | fail-close kept (precompute block) |
| `--seg-spike-reweight` | no | fail-close kept |
| `--seg-subpix-boundary-weight` | no | fail-close kept |
| `--seg-chroma-boundary-weight` | (cap 450) | fail-close kept |
| `--margin-weighted` (mw_active) | no | pre-existing twin non-support (not D15; v7-inactive) |
`_validate_logit_adjust_compat` narrowed to a documented NO-OP; the four genuinely-unrouted levers
keep their own precompute-block fail-closes; DSL `LogitAdjust` docstring updated (no longer "fails
closed").

## RSS waterfill projection (#294) — micro-batch × seed-islands × LADDER
Micro-batch's memory cost is the B× training-forward transient (B renders f0+f1 + B SegNet + B
PoseNet activations), SEPARATE from `witness_memory_preflight`'s verdict/cf/gt terms. MEASURED at the
#293 tiny test scale: serial 4.22 GiB → batched(B=4) 15.35 GiB (report-only). With wa-island (LADDER
amplify/persist) a SECOND batched SegNet forward over K frames adds ~1× the seg-activation term. At
n600 real scale the FIXED + cf-cache + gt terms dominate and the B× forward is a transient on top; the
#294 waterfill CORRECTLY keeps B **pinned to 1 (UNMEASURED)** until a MEASURED uncontended n600 curve
exists — `assess_micro_batch` refuses to invent a curve (guarded test kept green). Seed co-grad #293
composes: the dual value_and_grad differentiates the SAME `total_loss_fn_batch`; the offset is a
CONSTANT (no grad), the unify branch is differentiable — LADDER composition equivalence pinned by
`test_leg_logit_adjust_composes_with_wa_island_ladder`.

## What the council decides
1. Whether to run the n600 **trajectory A/B** (batched-fp-reduction gradient-noise profile) that makes
   micro-batch admissible for v7.1 — the routing is ELIGIBLE, not sufficient.
2. The micro-batch × seed-islands × LADDER **memory admission** (a MEASURED uncontended n600 forward
   curve to unlock the waterfill knob; without it B stays pinned 1).
3. (Hygiene, non-blocking) the #293 per-group seed tolerance (2e-4) sits below observed variance
   (3.4e-4) — a candidate tolerance re-fit, orthogonal to D15.
