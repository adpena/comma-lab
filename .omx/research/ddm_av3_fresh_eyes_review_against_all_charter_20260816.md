# ddm_av3 charter — fresh-eyes adversarial review AGAINST ALL of the 2026-08-16 signal arc

Operator authority: 2026-08-16 verbatim "That's a lot of really interesting signal perhaps its
worth doing another fresh eyes, review against all like we had been doing." Series successor of
av1/av2 (cross-model fresh-eyes adversarial passes). READ/AUDIT arm: no launches, no Modal, no
Metal, never write under /Volumes/APDataStore/pact/ddm_lr1/ (a live probe chain runs there).

## The arc under review (all landed 08-16 — verify at SOURCE, not from this summary)

1. **td1 → rt1** (memos `ddm_td1_token_drop_schur_arithmetic_20260816.md`,
   `ddm_rt1_seg_roundtrip_decomposition_20260816.md` §1–§7; commits a047fb1ad6, b4194d70fb,
   a0e3c8b9b3, ed699ffd5d, f59ac4f96e, f1a7a31d67, bdc54e01d5): seg axis = 33,743 flips 99.22%
   on the 1-px transmitted boundary, residual = TIE; coder gate PASSED (M7 32,270 B vs 35,117
   bar); η gate CLOSED the correction channel (η 0.6235 at n=9, 0-of-n above 0.753, NON-SUPPLIER
   0.0183 vs 0.0221 S); pose-aggregation sign error self-corrected (mean-of-ratios vs scorer's
   mean-of-d_pose); three self-caught defects total.
2. **b2e** (memo `ddm_b2e_edit_replay_admission_verdict_20260816.md`, commits f91e30904a,
   b7fa79a2fe; payloads /Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/):
   REGIME_THESIS_INSTANCE_REFUTED — 3 edits refused (collapse 0.75–1.06× vs ≥50× bar) BUT the
   F2-alone window barely trained (9 B weight delta at lr 2e-7 × 3000 steps).
3. **ns1** (memo `ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md`): FiLM blocks_1
   anisotropic sensitivity ~94×; §A screen Δd_pose ≤ 5.1e-9·ΔB; P1–P5 missing patterns.
4. **rc2** (memo `ddm_rc2_regime_charter_and_lr_probe_20260816.md`, commit d7efa7128f): three
   premise corrections (F2 edit-op ALREADY in-forward during b2e · #925 margin-weight is an
   MLX-witness PORT not a compose · F3 --film-row-dropout = the one never-fired lever) + the
   sealed ddm_lr1 probe ticket + the ddm_rg1 regime-charter draft (band_objective spec,
   vehicle = semantic trainer not wd3).
5. **lr1 in flight** (payloads /Volumes/APDataStore/pact/ddm_lr1/ — READ ONLY): A2 (lr 2e-5)
   trajectory 0.000286 → 0.000516 @step100 (+80%) → 0.000354 @step600 on ema_shadow evals;
   MAIN adjudicated MOVED-UPWARD → run-all-four; C0/A1/A3 landing during your run.
6. **The convergence claim itself** (task #1074/#1075 metadata + main_hot_state): "all post-hoc
   levers on all three axes are measured-bounded; every surviving route is training-regime work."
   This is the day's biggest claim — attack it directly.

## Seeded adversarial targets (audit these, then sweep beyond seed)

- **S1 EMA-lag confound on A2's trajectory** (the #85 bug class): step-0/100/... evals are on
  `evaluated_weights: "ema_shadow"` with EMA decay derived from 600-step run geometry (LawRef
  ema_decay_run_geometry_v1, trainer line ~921). Compute the actual decay + shadow lag at each
  eval point; quantify how much of "0.000286→0.000516 destruction then anneal-back" is live-weight
  movement vs shadow lag artifact. This changes what A1's verdict will MEAN — deliver before or
  with the A1 read if possible.
- **S2 rt1 η-gate scope**: is the closure SOLVER-family (sq1-adapted, described-set support, 25/…
  step budgets) or CHANNEL-family? Check §6.3's collateral law generality + the n=9 subset's
  license under m96 (seeded-random, may-refute rule). Was the pre-registered bar arithmetic
  (0.753 from §5 bytes) still the right bar after the coder-race numbers moved?
- **S3 mean-of-ratios sweep**: the newly banked law (memory
  pose_aggregation_is_mean_of_dpose_never_mean_of_ratios_20260816) — grep the CORPUS for other
  live consumers of per-pair ratio aggregation on pose or seg (tools/, experiments/, memos cited
  in live task rows). Any verdict resting on one is suspect.
- **S4 b2e diagnosis completeness**: is lr-starvation the WHOLE story of the 9 B window, or do
  frozen masks / 600-vs-3000 cosine geometry / EMA decay also bind? Verify rc2's three premise
  corrections AT SOURCE (editability_levers.py applied() in-forward; trainer argparse default lr
  line ~738; F3 never-fired claim).
- **S5 the convergence claim**: enumerate any lever NOT actually bounded by today's closures —
  e.g. td1's 807 label-correction sites (H3, priced-footnote status), rt1 §6.4's two named
  reopening conditions, mixed q3/q4 under sensitivity-AWARE allocation vs the §A screen's ~100×
  margin, the wd3 warm-lineage rate rung, the js8 pose line's actual next row. Over-scoped
  convergence = the same stale-headline genus the day already hit twice.
- **S6 instrument coherence**: quantized_exact_seg (in-loop, vs transmitted tokens @384×512) vs
  the advisory archive instrument (vs GT labels) vs contest d_seg — is the lr1 bar's 1e-5
  absolute anchor honest, and is the step-0 0.000286 vs advisory 0.000427 gap understood or a
  red flag?
- **S7 beyond-seed**: stale headlines vs corrected bodies across today's memos/task rows;
  unconsumed follow-ons (fire-at-harvest law); anything measure-and-discard shaped.

## Deliverables

Ranked findings table (severity × what-it-changes × verified-at-source evidence), corrections
applied AT SOURCE where cheap (headline+body+ledger, per the stale-headline law), routing
implications for (a) the lr1 adjudication, (b) the ddm_rg1 charter, (c) the convergence claim.
Memo `.omx/research/ddm_av3_fresh_eyes_review_20260816.md` committed via
tools/subagent_commit_serializer.py with --expected-content-sha256; verdict_scope on every
negative; final message persisted with NEXT_IF_RESUMED; done receipt via the keeper.

## OPTIMAL FORM

Reference form: the av1/av2 adversarial-pass family (fresh-eyes, verify-at-source, refute-first).
This is the declared 08-16-arc instance. SCOPE: today's arc + its direct premises; MECHANISM
unreduced (full source verification, no summary-trusting). TOY-BRACKET: none.
