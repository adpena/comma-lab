# v6 spec — 4-arm composite ablation (partner-amended; PRE-REGISTERED before run)

UTC: 2026-06-06T23:30:00Z · Status: DESIGN (no code edits until the triple-audit
verdict on swarm results lands; v6 consumes swarm-B's composite operator and
swarm-F's pose threading, so both must clear audit or be remediated first).

## Question v6 answers

v4: birth works pose-blind. v5: pose guard live → rejected 3/3. v6 tests the
ALGEBRA: does frame1 SegNet birth + frame0 PoseNet compensation, admitted by
EXACT nonlinear score, produce one accepted composite evaluator action?

## The cap-scale finding v6 must resolve (quantified)

v5's raw guard caps POSE-OUTPUT movement: ||p_new − p_initial||₂ ≤ 0.05 (6-dim
head). At the v5 operating point d_pose ≈ 194 (batch-local MSE over dims):
‖p_old − target‖₂ ≈ √(6·194) ≈ 34.1, so the cap bounds Δd_pose ≤
(2·34.1·0.05 + 0.05²)/6 ≈ 0.57, i.e. score cost ≤ (5/√(10·194))·0.57 ≈ 0.065
batch-local units. The same smoke's worst region carried ~22.4 batch-local seg
units of debt. A raw cap that rejects before the exact joint gate can weigh
22.4 against ≤0.065 is structurally over-rejecting at this operating point —
exactly the partner's "mis-scaled at d_pose≈194". v5 likely over-rejected;
`would_accept_exact_score_if_raw_cap_disabled` will measure it.

## Admission semantics change (lands post-audit, before v6)

1. PRIMARY: exact ΔS_nonrate < −ε (batch-local; the evaluate.py functional).
2. CATASTROPHIC guard (kept, demoted): reject if
   d_pose_new > d_pose_old·(1 + catastrophic_relative_cap) OR
   pose_score_regression > allowed_pose_score_regression (score-scaled), with
   a generous absolute hard_cap retained as backstop.
3. Per-step rejection rows persisted (v5 gap: only counters exist):
   {step, old/new_d_pose, Δd_pose, pose_score_delta, seg_score_delta,
    exact_delta_score_nonrate, raw_pose_cap_result, exact_score_result,
    hard_won_count, would_accept_exact_score_if_raw_cap_disabled,
    rejected_by_raw_pose_cap | rejected_by_exact_delta_score |
    rejected_by_catastrophic_pose_guard}.

## The 4 arms (same pair/region, same seed, real video, real teachers)

A. frame1 hard-birth only — exact-score admission.
B. frame0 pose-compensation only (no birth) — isolates the free axis.
C. independent composition: accepted-A then accepted-B, composite re-priced.
D. joint line search: birth step + compensation inside one admission loop
   (swarm-B's operator path).

Each arm emits ONE ActionEffect row (tac.action_effect.v1): old/new d_seg +
d_pose, exact delta_score_nonrate, raw pose delta, pose score delta,
transitions (wrong_to_target / target_to_wrong / wrong_to_wrong /
net_target_support_delta), uint8_changed_count_region,
seg_input_delta_linf_region, posenet_input_delta_linf_pair,
authority=batch_local_live_mlx, promotion_eligible=false, plus the
counterfactual/rejection fields above.

## Gate refusal conditions for v6 (the partner amendment, adopted)

validate_nerv_long_run_gate must refuse a v6 receipt unless: exact-score
admission semantics present (per-step rows + counterfactual fields), pose
trust semantics present (pose_guard.available + contest resolution), transition
disambiguation present, and the arm rows serialize to ActionEffect. (Gate v2
check additions queued post-audit.)

## Sister asks closed by existing landings

- action_id continuity across L4 surfaces: ENFORCED (gate, landed).
- transitions in receipts: LANDED (c3e9bb1b8).
- CUDA/T4 drift screen for top-N frontier candidates: in roadmap; PR110's
  CPU/CUDA split on identical bytes is the anchor.
- Canonical ActionEffect pair (one HiNeRV v6 row + one PR110 replay row):
  queued immediately post-audit — the "one shared currency" existence proof.

## v5 rejection-row table (honest): NOT EXTRACTABLE retroactively

v5 persisted counters only (pose_guard_rejected_step_count=3), not per-step
old/new d_pose rows. The instrumentation in §3 exists precisely so v6 emits
the table the partner asked for; fabricating a v5 table is forbidden.

## Amendments accepted from GPT review (pre-run; 2026-06-07)

1. **Dual epsilon.** `epsilon_research_batch_local = 1e-6` for v6 live rows
   ONLY (labelled research-stage numerical epsilon). L4/L5 (fakequant /
   parse-back / inflate) admission uses a replay-calibrated epsilon:
   `max(1e-5, 5 × measured_replay_noise_floor)`; the noise floor itself must
   be measured and reported before any survival-row admission reuses ΔS.
2. **Catastrophic guard tightened + counterfactualized.** Primary admission:
   `exact_delta_score_nonrate < -1e-6`. Backstop (blowup-catcher, NOT a
   tradeoff policer): `pose_score_regression ≤ 0.5 batch-local` AND
   `d_pose_new ≤ 1.25 · d_pose_old`. Both counterfactuals emitted per step:
   `would_accept_exact_score_if_raw_cap_disabled` AND
   `would_accept_without_catastrophic_guard` (at d_pose≈194 the ratio cap
   permits ~9.9 score units, so the 0.5 score cap dominates — the logs must
   show whether the ratio cap ever binds).
3. **Interaction/commutator field.** v6 emits
   `interaction_or_commutator = ΔS(C) − ΔS(A) − ΔS(B)` — the first HiNeRV
   row in the action-commutator ledger, connecting the servo directly to the
   PR110 algebra lane (composite deltas ≠ sums under scorer nonlinearity).
4. **Decision matrix deliverable.** Post-run table: A accepted/rejected-by;
   B pose-improved/seg-unchanged; C exact ΔS + interaction; D exact ΔS +
   improvement-over-C. Interpretation contract: A-accepts-but-old-cap-rejects
   ⇒ v5 was a cap failure; only-D-accepts ⇒ future operator is joint line
   search; no-arm-accepts ⇒ first failing surface is the next patch target.
5. **Remediation namespace.** If the audit returns confirmed-HIGH on any v6
   dependency (B operator / F threading / D schema), v6 proceeds only after
   remediation AND every v6 `action_id` includes a
   `remediated_dependency_hash` component so pre-audit rows can never be
   compared with post-remediation rows.
6. **Must-have field list** (superset of §"The 4 arms"): action_id, arm,
   old/new d_seg + d_pose, seg/pose score deltas, exact ΔS_nonrate, old/new
   raw region debt, all four transition counts + net, hard_won_count,
   argmax_changed_count_region, pose_output_l2_delta, pose_input_delta_linf,
   raw_cap_decision, exact_score_decision, catastrophic_guard_decision, both
   would_accept counterfactuals, restore_state_pass,
   authority=batch_local_live_mlx, promotion_eligible=false.

## Taint rule (reinforced, in force)

No swarm output (B operator, F threading, A/C/D/E modules) updates launch
evidence or gets consumed by v6 until the 18-skeptic audit verdict is
committed; any confirmed-HIGH target is remediated + its receipts reframed
before use.
