# ddm_jo1_joint_objective_design — exact joint objective and rc2 build receipt

Date: 2026-08-21  
Task: `ddm_jo1_joint_objective_design`  
Verdict: **BLOCKED**  
Axis: design/build receipt only; no scorer authority and no Modal dispatch  
Vehicle: rc2, archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`

## Result

The exact joint objective, typed configuration, stage-boundary acceptance,
checkpoint/retention contract, single-`p` package validator, and MAIN-owned
dispatch gate are built and tested. They do not constitute a runnable trainer.
The build fails closed on five named blockers:

1. `RC2_FRESH_SCHUR_RECEIVER_CLOSE_NOT_IMPLEMENTED`;
2. APDataStore had 33,404,878,848 B (31.111 GiB) free at seal time, below the
   inherited 47,244,640,256 B (44 GiB) retention preflight by 12.889 GiB;
3. the full rc2 n600 T4 base argmax field was not retained;
4. the n600 source Pose6 target payload was not retained;
5. no fresh real-configuration T4 peak-memory receipt exists.

No scorer, trainer, exact evaluator, or Modal job ran. No candidate archive was
created. The rc2 pointer did not move.

## RECALL EVIDENCE

### Scopes and queries

- Full research corpus: content searches for `collateral`, `introduced`,
  `fixed`, `B/H`, `receptive`, `stride-2`, `Schur`, `realized acceptance`,
  `joint remeasure`, `single p`, `rc2 argmax`, `pose6`, and the old `ddm_jo1`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py
  --json`, then score-marginal, task-space variational, shared-resize, and
  realized-acceptance equations.
- Graph/state: `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks,
  `.omx/state/main_hot_state.md`, `canonical_task_status.jsonl`, and current
  lane-claim summaries.
- Implementation: EC2 worker/dispatcher, JG1/JG5/v19 realized acceptance,
  QS1/QS5 exact-object compensation, JS1B field materializer, WD3 strict config
  and resume, HR1 memory receipt, rc2 runtime/package parser, and the evaluator.
- Custody: bounded APDataStore/VertigoDataTier searches for rc2 argmax fields,
  decoded tokens, and Pose payloads. No global nonexistence claim is made.

### Findings beyond the charter seeds that changed the plan

1. QS4's exact six-pair decomposition put 60 of 76 introduced cells at
   neighboring sites and only 16 at the edited site. Collateral is receptive-
   field spill, so a same-site mask is not a sufficient constraint.
2. JG1 showed frame-1 edits can increase pose damage by 104.6–822.7× before an
   exact-object carrier re-solve, while re-solving all 12 carrier coefficients
   recovered 98.7–100%. Fresh same-object compensation is therefore part of
   compile feasibility, not a later patch.
3. CF2 measured direction/set-dependent token rates: 3.8373 bits/token for the
   shipped set versus 5.9467 bits/token for marginal additions. A modeled token
   rate is only a ranker; every stage needs an actual rc2 re-encode.
4. JS8's post-hoc Road-hub gate fixed only seven flips for +1,749 B, while its
   review kept gate-aware training live. This ruled out adding another post-hoc
   gate and favored training with the correct-cell closure active.
5. v19 realized acceptance only sums coordinates after independence is shown.
   A shared conditioner is coupled, so JO1 must remeasure the cumulative object.
6. `ddm_jo1` already names a distinct six-event CP135 object whose measured row
   was worse by +0.000216 through pose. New custody is therefore isolated under
   `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/`.
7. The latest bounded lane summary had no active EC2 claim. That removes the
   stale “four active claims” blocker, but any future job still needs a new MAIN-
   owned unique claim at its fire trigger.
8. The exact named memory hook
   `seg_mechanisms_die_on_collateral_not_targeting_20260821` was not found in the
   bounded memory registry/repository search. No memory claim was fabricated;
   primary receipts were used.

The common contract's older frontier paragraph was not used: live authority is
the later rc2 row in `main_hot_state.md` and the rc2 exact receipt.

## Derived objective

Let (P=600\cdot384\cdot512=117,964,800), (D=37,545,489),
(B_0=180,456), and (p_0=6.370359\times10^{-6}). For exact parsed archive
(A), define:

- (F(A)=\#\{b\ne g, c(A)=g\}), fixed rc2 errors;
- (H(A)=\#\{b=g, c(A)\ne g\}), introduced errors;
- (W(A)=\#\{b\ne g,c(A)\ne g,c(A)\ne b\}), wrong-to-wrong changes,
  reported with no fake credit or duplicate penalty;
- (p(A)), first-six PoseNet MSE on the same parsed render;
- (B(A)), actual ZIP bytes after the real coder.

The exact realized action is:

\[
\Delta S(A)=
{100(H-F)\over P}
+\sqrt{10p(A)}-\sqrt{10p_0}
+{25(B(A)-B_0)\over D}.
\]

Admission requires all of:

\[
p(A)\le6.370359\times10^{-6},\qquad
H/F\le0.89,\qquad
\Delta S(A)<0,
\]

plus deterministic single-member-`p` parse-back, deterministic repeat, and a
fresh compensation fingerprint bound to the exact compiled object. If (F=0),
any (H>0) violates the collateral cap; (F=H=0) has ratio zero but is still
rejected by nonnegative complete delta.

### Units and measured constants

| Term | Value | Units | Provenance |
|---|---:|---|---|
| Seg marginal (100/P) | `8.4771050347e-7` | score / net flip | evaluator-derived |
| Rate marginal (25/D) | `6.6585895312e-7` | score / archive byte | evaluator-derived |
| Exchange | `1.2731082153` | bytes / net flip | derived from the two marginals |
| Pose marginal at rc2 | `626.4700138` | score / unit d_pose | `5/sqrt(10*p0)` |
| rc2 errors | `23,757` | pixels | contest-CUDA T4 n600 receipt |
| repinned adapter price | `1,176` (`1,174..1,191`) | actual rc2-coder bytes | EC2P coder measurement; not receiver-closed |

The allowed (3.59\times10^{-10}) pose rise costs
`2.248995663e-7 S`, only 0.2653 flip-equivalents or 0.3378 byte-
equivalents. Pose must therefore be effectively invariant.

## Differentiable solve and stage authority

For GT winner-rival margin (m_i) after exact (R), the inner solve uses one
full-field denominator for both relaxed populations:

\[
F_\tau=P^{-1}\sum_{i:b_i\ne g_i}\sigma(m_i/\tau),\qquad
H_\tau=P^{-1}\sum_{i:b_i=g_i}\sigma(-m_i/\tau).
\]

Using separate conditional means is forbidden: it would give the small error
pool the same aggregate mass as roughly 118 million correct cells and erase the
measured collateral physics. The constrained surrogate is:

\[
\mathcal L=
-F_\tau+H_\tau+r_{proxy}
+\eta[H_\tau-0.89F_\tau]_+
+{\kappa_c\over2}[H_\tau-0.89F_\tau]_+^2
+\nu[p-p_{max}]_+
+{\kappa_p\over2}[p-p_{max}]_+^2.
\]

`rate_proxy` is proposal-only. Real coder bytes enter only at each exact stage
boundary. Duals update only after the parsed full-n600 field pass; rejected
stages roll back. Every stage retains live and EMA candidate fields, B/H/W,
Pose6, actual archive and deterministic repeat, parsed-render identity, metrics,
optimizer/RNG/dual state, and field/package cursors.

This is collateral inside the solve: the previous realized stage's full-field
violation changes the next stage's dual state. It is not end-only measurement or
post-filtering.

## Actuation adjudication

| Form | Evidence | Verdict |
|---|---|---|
| EC1 conditioning only | 12,075 fixes (34.53% gross) but 52,854 introduced, H/F 4.377 | old fixed-ramp/end-only formulation dead; family not closed |
| direct/additive only | QS4 H/F 0.7037 on six pairs, but QS5 only 17 net fixes on its three-pair support; additive sidecar costs 84.48 B/flip | locality prior live; standalone demonstrated mass/rate insufficient |
| hybrid | oriented token context proposes; bounded RGB actuation occurs after all renderer TokenBlocks and before exact R; full correct-cell and Pose constraints decide every stage | selected design hypothesis; unmeasured on rc2 |

EC1 injects before four nonlinear 3×3 TokenBlocks with dilations 1,1,2,4; its
conditioner also has neighbor context and two 3×3 convolutions. A perturbation
there spreads through the renderer before it reaches SegNet. The selected output
actuation removes those four renderer blocks from the propagation path.

It cannot remove the irreducible physics: SegNet's EfficientNet-B2 stem is a
3×3 stride-2 convolution, so adjacent RGB sites enter overlapping stride-phase
neighborhoods before deeper U-Net mixing. The measured QS4 neighbor harm is the
empirical signature. JO1 suppresses this bleed by constraining all base-correct
cells in the receptive-field closure and measuring the entire field, not by
claiming pixel locality.

The implemented actuator/objective is a tested primitive, not a runnable rc2
trainer. Folding its learned content into the rc2 RX1 `p`, updating section
lengths/runtime pins, executing the exact receiver, and performing the fresh
same-object carrier/Schur re-solve remain unimplemented and are the primary
named blocker.

## Preregistration and arithmetic correction

The operator's labels are preserved exactly:

| Label | Net fixed flips |
|---|---:|
| LIVE | `>=965` |
| MARGINAL | `924..964` |
| CLOSED-neutral | `0..923` |
| CLOSED-harmful | `<0` |

At +1,176 B, continuous break-even is `923.7235184` flips, so integer
break-even is 924. The preregistered 965 boundary gives
`DeltaS=-3.4990507e-5` before pose, 9.9973× rather than strictly 10× a
`3.5e-6` band. Strict 10× begins at 966 flips. The labels remain unchanged;
their arithmetic meaning is now explicit.

The prior-law arithmetic also required correction:

\[
F_{prior}=23,757\cdot{12,075\over34,970}=8,203.1963.
\]

| Net target | Required H/F | Suppression from measured 52,854/12,075 |
|---|---:|---:|
| 924 | `0.887360979` | `4.932764636x` |
| 965 | `0.882362927` | `4.960705761x` |
| 966 | `0.882241024` | `4.961391207x` |

Thus the charter's rounded `4.93x => >=965` equivalence is false: 4.93× predicts
about 919.9 net flips. `H/F<=0.89` and “4.93×” remain registered operational
caps; a mathematically supported LIVE prediction at transferred gross recovery
requires about 4.96071× suppression.

The FORMULATION-level rc2 falsifier remains: if optimal-form constrained
training reduces gross recovery below 3.9% of the 23,757-error pool, it cannot
pay the 1,176 B anchor even at zero collateral. If collateral is still falling
at the final fail-safe cap, that is a stopping-rule defect, not a family close.

## Build and custody status

| Component | Status | Evidence / boundary |
|---|---|---|
| strict typed config + provenance | BUILT | unknown fields refused; arrays declare shape/dtype; source hashes pinned |
| exact delta/bands/admission | BUILT | pure tested functions use realized B/H/Pose/archive bytes |
| joint surrogate | BUILT-PRIMITIVE | common full-field denominator and backprop tested; not bound to rc2 scorers |
| stage field-pass config | BUILT | all three stages force `at_end=true`, start 0, count 600, retained outputs |
| atomic checkpoint contract | BUILT | distinct live/EMA/optimizer/RNG/dual/cursors retained with SHA/bytes |
| single-`p` package validator | BUILT-VALIDATOR | repeat/parse-back/fingerprint gates exist; no candidate built |
| rc2 decoded semantic tokens | RETAINED INPUT | 117,964,800 B, SHA `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| contest-CUDA GT argmax | RETAINED INPUT | `(600,384,512) uint8`, SHA `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` |
| rc2 base argmax | BLOCKED | did not find a retained field in bounded rc2/AP/Vertigo searches; T4 row is scalar-only |
| source Pose6 targets | BLOCKED | no JO1-bound source target payload retained |
| real-scale T4 memory | BLOCKED | 16 GiB request is allocation, not measured peak; no receipt |
| AP storage | BLOCKED | 31.111 GiB free versus 44 GiB required at seal r4 |
| rc2 receiver-close + fresh Schur | BLOCKED | implementation intentionally refuses before claim/spawn |
| overall | **BLOCKED** | not READY_TO_FIRE |

### Retained seal r4

All new receipts are on APDataStore, not `/tmp` or Vertigo:

| Payload | Bytes | SHA-256 |
|---|---:|---|
| `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/seal_r4/compiled_config.json` | 13,397 | `deb07cd1a312c53b9b0275d02ae74c102c2ae486075f442f62c67a4b95b0c8fa` |
| `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/seal_r4/READINESS.json` | 629 | `ffe86276b325089b72a34c12664984ef256064fcc599ada538d4133f89527d70` |
| `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/seal_r4/FIRE_ORDER.json` | 1,131 | `caae3a14fd8e16b3e6846cac665cd17a1e4a9a6a054ae3633c732d72244008ef` |

The compiled workload identity is
`ef3134ce1a60188912463dc213f5d79c5d3db1d6ba9bdde5f2f551a9c88a72ab`.
The seal binds the exact rc2 archive/runtime, decoded token payload, GT field,
source object, frozen scorer weights, and all three JO1 Python sources.

## Fire disposition

Disposition: **QUEUED-WITH-FIRE-ORDER, currently BLOCKED**. MAIN owns dispatch,
and the arm did not claim or fire a lane.

The first governed command, already sealed, is:

```bash
.venv/bin/modal run experiments/ddm_jo1_modal_joint_objective.py::materialize_scorer_payloads \
  --compiled-config /Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/seal_r4/compiled_config.json \
  --expected-config-sha256 deb07cd1a312c53b9b0275d02ae74c102c2ae486075f442f62c67a4b95b0c8fa \
  --main-owned-dispatch-authorization --detach --provider-detach-ack
```

Its fire trigger is **not yet satisfied**: the receiver-close/materializer
backend must first be implemented, reviewed, and remove the hard blocker. The
current entrypoint refuses before claim or spawn. After harvest, MAIN must stop,
bind the retained base/GT/Pose payload hashes into a newly sealed config, and
only then emit the exact memory-preflight command. After that harvest, MAIN must
stop and reseal again before training. Null successor commands in the fire order
prevent a stale three-step fire.

## Verification

- Review pass 1 (`jo1_pass1_final_math_custody`) checked exact arithmetic,
  shared-denominator collateral scaling, artifact shapes/hashes, AP-only output,
  checkpoint completeness, and stale-command prevention. It corrected rounded
  evidence previously labeled exact and wired stage weights into the primitive.
- Review pass 2 (`jo1_pass2_no_fake_operations`) challenged the shared assumption
  that a learned output residual is already a runnable vehicle. It is not: no
  real-scale memory or scored quantity was measured, so the hard implementation
  blocker remains in readiness and every entrypoint refuses before lane claim.
- JO1 focused suite: **11 passed**, 2 non-failing Pydantic field-shadow warnings.
- `py_compile`: passed for all four JO1 Python files.
- `ruff check`: passed for all four JO1 Python files.
- Developer preflight: 17/25 gates green, 8 red. Bounded follow-up attributed
  every red gate outside the five JO1 files: one existing strict-state loader,
  one existing custody tag site, one existing shared-state writer, 25 legacy
  ad-hoc launch surfaces, an AGENTS claim-closure documentation gap, 124 old
  landing memos, 14 pre-existing lane tokens, and five existing substrate loss
  adapters. No waiver was added and no unrelated surface was changed.
- Wider non-scorer regression: **142 passed, 4 failed**. The failures are outside
  JO1's files: two JS1B fixtures conflict with a newer hard custody hash, the old
  unrelated JO1 probability-object test timed out hashing a Vertigo payload, and
  the shared resume-registry test sees an unrelated unregistered
  `_g111_lineage_state_arrays` producer in the dirty worktree. No failing stack
  enters the new JO1 design/worker/dispatcher/test files.
- No full scorer, Modal call, training, archive creation, or exact evaluation was
  performed.

## LIVE-HYPOTHESES

- The post-TokenBlock hybrid can retain conditioner-scale gross recovery while
  full-field collateral duals reduce H/F to at most 0.88236 for LIVE arithmetic.
- Full stage-boundary fields may expose an earlier low-collateral checkpoint
  hidden by EC1's end-only measurement.
- Fresh exact-object carrier/Schur resolution may satisfy the severe rc2 pose
  cap after a render-side change.
- The rc2 single-`p` grammar may carry the learned residual and compensation
  payload at a price close enough to the 1,176 B coding anchor; this is not yet a
  receiver-closed measurement.

## DEAD-ENDS

- Re-firing the CP135 EC2 order or reusing its fields/tokens/compensation.
- Original fixed-ramp, end-only EC1 instance: collateral-negative.
- Post-filtering harmful pixels or measuring the field only at the endpoint.
- Global deblur/post-hoc RGB correction: family-closed on pose.
- Additive Seg sidecars at this density: rate-dominated.
- Three-pair direct micro-edits as the whole solution: measured 17-flip ceiling.
- Borrowed or stale Schur compensation and modeled token rate as admission.
- A second ZIP member, a non-retained payload, or a local/macOS field promoted as
  contest-CUDA authority.

Own-vehicle frontier: **rc2 S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**, archive `df7fd266…`; **UNMOVED**.
