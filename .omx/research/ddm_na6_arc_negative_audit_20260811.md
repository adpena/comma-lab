# ddm_na6 — negative-results audit of the 08-10→08-11 arc

**Date:** 2026-08-11
**Lane:** `ddm_na6`
**Mode:** scorer-free, read-only on live solve stores, no Modal
**Frontier authority at audit:** composed pointer `S=0.16195513827824176 @ 186,252 B`
`[contest-CUDA T4,n600]`; own-vehicle LC2 frontier
`S=0.16959899569230852 @ 187,226 B [contest-CUDA T4,n600]`.

## Verdict

The arc's negative-result discipline survives this audit. Of 22 non-overlapping scientific
closure rows, 18 are **UPHELD**, 4 are **RE-SCOPED**, and 0 are **REOPENED w/ $0 probe**. The four
scope corrections are:

1. LC2's CPU↔CUDA score gap is an `INSTANCE(lc2 exact bytes)` result, not a vehicle-family law.
2. The old ps135 Stage-C materialization blocker is an `INSTANCE(old landed driver/hash)` result;
   GEN-2 cleared q4 parity and cold-store relocation, while the sequential scorer/master binding
   remains a real gate.
3. The rejected projected-global ps135 start is an `INSTANCE(whole projected candidate)` result;
   row-local projected starts are positive evidence and won 95 selections in GEN-2 pass 1.
4. PE3's explicit-label substitution negative and sr1's entropy-context negative do not occupy
   the same claim cell. Their earlier “convergence” language is narrowed: explicit target
   substitution is closed, while joint distortion-side conditioning remains open and owned by
   JS1/#995.

This is 18.2% re-scoped and 0% reopened. It falls inside the pre-registered 2–4 re-scope / 0–1
reopen prediction. Neither process falsifier fired: reopened share is not above 30%, and the audit
did not return zero re-scopes.

No scorer or materializer was run. No payload was materialized. The live ps135 store was only
read. Its newest complete receipt at audit time is pass 2:
`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/leg_a/passes/pass_02/receipt.json`, with 517
accepted rows, `187,221 B`, archive SHA-256
`b8c322875462171ba1043c0ab6a5a8e32ff5ff9578c8a29cf3b469a649d9b24a`,
`d_seg=0.0004273478190104167`, `d_pose=0.000014717098986238853`, and recomputed
`S=0.17952896607020802 [macOS-CPU advisory,n600]`. Pass 3 was non-terminal, so none of its
partial state is used as evidence.

## Scope and counting rule

The denominator is 22 mechanism-level closure rows: 18 from the seven pinned arc memos and four
standing/crosswalk rows. Closely related implementation attempts are consolidated only when they
share the same claimed mechanism, evidence object, scope, and consumer. Temporary custody states
such as “do not score now,” local scorer-lane ownership, or an in-progress pass are not scientific
negatives and are listed separately rather than inflating the denominator.

`FAMILY` below means only the explicitly named mechanism family. It never means all conditioning,
all learned coding, all pose solving, or all native acceleration.

Authority convention for numeric rows: exact rate, parse-back, bank-fit, and receiver figures from
the scorer-free arc receipts are `[macOS-CPU scorer-free,n600]` unless a row names another
population; ps135 search counts and scores are `[macOS-CPU advisory,n600]`; runtime attribution is
`[macOS-CPU runtime]`; PE3 is `[macOS-CPU advisory,stratified-random n32]`; #881 is
`[macOS-CPU advisory,matched n74]`; contest axes are written explicitly. These labels classify the
measurement surface and do not promote advisory evidence.

## Re-grade table

| ID | Negative or close, with receipt | Stated scope | Instrument validity in the claim's units | Re-grade | Consumer / disposition |
|---|---|---|---|---|---|
| A1 | sr1 causal edge context: charged stream `114,704 B` vs `114,706 B`, but held-out net `−143.491 bits` after table cost (`.omx/research/ddm_sr1_implicit_edge_conditioning_20260811.md`) | `FORMULATION(F26 additive causal edge context)` | Valid. Full n600 token likelihood and serialized charged bytes measure the rate claim directly; the held-out sign is more negative than the headline. | **UPHELD** | JS1/#995 receives only the still-open distortion-side joint-conditioning residue; `QUEUED-WITH-A-FIRE-ORDER` in the live task graph. |
| A2 | sr1 scalar pose sign/delta context costs `+43 B` (`ddm_sr1_implicit_edge_conditioning_20260811.md`) | `FORMULATION(F26 additive scalar pose context)` | Valid. Full n600, same token stream and charged serializer as the baseline. It measures rate, not distortion. | **UPHELD** | Folded into sr1. Do not retry another small post-hoc scalar context on F26. |
| A3 | Checkerboard/group reordering carries no new information; CPC1 exact causal partition costs `255,288 B` vs `114,706 B` (`ddm_sr1_implicit_edge_conditioning_20260811.md`) | `INSTANCE(reorder-only)` and `FORMULATION(CPC1 exact partition replacement)` | Valid for serialized rate on the full F26 lattice. Neither instrument measures a learned joint renderer. | **UPHELD** | Folded into sr1; joint conditioning stays with JS1/#995. |
| A4 | Explicit PE3 label/contour/mask substitution worsened all 32 stratified frames and charged `74,408 B` (`.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md`; cross-cited by sr1) | `FORMULATION(explicit target substitution on LC1/PE3)` | Valid for realized-through-R segmentation and charged bytes: stratified non-prefix n32, exact receiver. It does not test conditioning-only. | **UPHELD** | Explicit substitution is folded closed; conditioning-only is handled in S4 below. |
| A5 | PGQ1 ranks ≤5 fail the exact-bank gate; best rank 5 bank MSE `5.202855e-6` exceeds `2.5e-6` (`.omx/research/ddm_pz4p_pose_gauge_preproof_20260811.md`) | `FORMULATION(PGQ1 on current LC2 bank)` | Valid. All 600 bank rows are represented, the error is measured in the coefficient-bank gate's units, and the memo does not transfer the result to all low-rank codecs. | **UPHELD** | Folded into pz4p. Rank 6 remains the only admitted PGQ1 envelope point. |
| A6 | Frozen post-hoc PK3 gauges save only `64 B`, `1,936 B` short of the gate; low-rank plus exact residual costs `+4,316 B` (`ddm_pz4p_pose_gauge_preproof_20260811.md`) | `FORMULATION(frozen post-hoc gauge)` and `FORMULATION(PGQ1+exact residual)` | Valid. Full-bank byte accounting is in the rate claim's units. It makes no scorer claim. | **UPHELD** | Folded into pz4p; no frozen-gauge retry. |
| A7 | `r6_b12_global`'s `168,005 B` preproof envelope is non-renderable; direct-v6 realizes `183,137 B`, only `−4,089 B` vs LC2 (`.omx/research/ddm_pz4p_pose_gauge_preproof_20260811.md`, `.omx/research/ddm_pz4r_pgq1_receiver_20260811.md`) | `FORMULATION(PGQ1 envelope as rendered candidate)` | Valid. The claim is parse-back/receiver realizability and exact archive bytes; direct-v6 is receiver-closed. Scorer effect remains pending and is not inferred. | **UPHELD** | Direct-v6 exact scoring remains `QUEUED-WITH-A-FIRE-ORDER` for the pz4r scorer successor after ps135 releases the lane. |
| A8 | PZ3-weight reuse, exact coefficient residual, direct SE(3) warp of six PoseNet outputs, and the 12-bit modular metric all fail their stated receiver/representation gates (`ddm_pz4r_pgq1_receiver_20260811.md`) | `INSTANCE(PZ3 reuse)`, `FORMULATION(exact residual)`, `FORMULATION(direct output warp)`, `INSTANCE(12-bit metric)` | Valid for the named parse-back, rate, or representation claim. The withdrawn modular f8 number is not used as evidence. | **UPHELD** | Folded into pz4r. The recorded winner substitution is `r6_b12_global → target_quadratic` for realized direct-v6 because the envelope winner was not renderable; this is present in the pz4r receipt and is not a missing-record finding. |
| A9 | Same-state F26 ANS loses to RC64 by `+6 B` (control) and `+9 B` (HP3) (`.omx/research/ddm_lp135_lossless_pack_20260810.md`) | `INSTANCE(F26 same-state ANS on two pinned streams)` | Valid. Exact encoded payload sizes and decode equality measure the lossless rate claim directly. | **UPHELD** | Folded into lp135/cp135. Keep RC64 for these sections. |
| A10 | The supposed ≥1 KB FD135 residue is absent: all `4,328 B` are already banked in CP135; CAP1 then saves only `79 B`, and direct CAP1→LC2 composition has a representation mismatch (`ddm_lp135_lossless_pack_20260810.md`) | `INSTANCE(CP135 ancestry)`, `FORMULATION(CAP1 exact field pack)`, `INSTANCE(direct cross-representation apply)` | Valid. Exact ancestry and archive manifests answer the residue claim in bytes; the mismatch is not generalized to all composition. | **UPHELD** | Folded. Any future rate arm must name bytes outside the already-banked FD135 object. |
| A11 | CP135 coder races: SMEVR wins `0/14`; shared LOTTO dictionary costs `+136 B`; supermask costs `+254 B` (`.omx/research/ddm_cp135_rate_compose_20260810.md`) | `FORMULATION(tested candidate set on pinned section objects)` | Valid. Exact payload/archive bytes on the composed candidate; no score extrapolation. | **UPHELD** | Folded into cp135. Do not promote this to a theorem about all learned or shared coding. |
| A12 | Native entropy decode takes only `1.1–3.1 s`; over 99.5% of wall time is HPAC probability generation (`.omx/research/ddm_rc64p_native_cpu_decode_20260810.md`) | `INSTANCE(LC2 Route-A shipped path on measured host)` | Valid. Wall-clock attribution measures the CPU-runtime claim. It does not test native lowering of probability generation. | **UPHELD** | Native entropy-only cure folded closed; C/Rust lowering or parallel restructuring of probability generation remains open to a runtime owner. |
| A13 | Cached-plan HPAC is exact but does not produce a timing win (`ddm_rc64p_native_cpu_decode_20260810.md`) | `INSTANCE(cached-plan HPAC implementation)` | Valid for that implementation and host timing; invalid as a family closure, which the source does not claim. | **UPHELD** | Folded into rc64p. |
| A14 | Exact LC2 bytes score `S=0.20728492781521812 [contest-CPU,n600]` vs `0.16959899569230852 [contest-CUDA T4,n600]`, with pose 6.6× and seg 1.45× worse on CPU (`ddm_rc64p_native_cpu_decode_20260810.md`, ADD.2) | Previously easy to read as a vehicle/device law | Valid for the identical LC2 archive bytes on the two measured contest axes. Invalid for transferring the magnitude or sign to ps135 output. | **RE-SCOPED** to `INSTANCE(lc2 exact bytes; two measured devices)` | Endpoint owner MAIN must replay the identical terminal ps135 archive on both contest-CPU and contest-CUDA before pointer action; `QUEUED-WITH-A-FIRE-ORDER`. |
| A15 | PR135's exact ±1 singleton sequence exhausts at `412,187,72,39,15,9,2,0`; PR133 integer-code copy, #740 direct CPR1 solve, and #460 n1 precision transfer are closed (`.omx/research/ddm_ps135_pose_resolve_20260810.md`) | `FORMULATION(current PR135 F26 ±1 neighborhood)` plus named `INSTANCE`/representation transfers | Valid. Full-population sequential scorer refresh makes the singleton claim in exact score units; #740/#460 are correctly rejected for vehicle/instrument mismatch rather than converted into family negatives. | **UPHELD** | Exact singleton neighborhood folded closed. Radius-2/multistart is a distinct admitted search and is already firing in ps135. |
| A16 | Uncompensated mixed precision is not a Stage-C candidate; smaller SD1M archives are rate artifacts, not score wins (`ddm_ps135_pose_resolve_20260810.md`) | `FORMULATION(uncompensated precision)` and claim-boundary close | Valid. The first omits the mechanism required for correctness; the second lacks exact-score evidence. Neither is a family result. | **UPHELD** | Folded into ps135. Keep compensation and exact scorer selection mandatory. |
| A17 | Stage-C bank materialization at the old landed driver/hash was blocked by q4 parity and scorer/master seams; GEN-2 later proves exact q4 parity and valid cold relocation (`ddm_ps135_pose_resolve_20260810.md`, GEN-2) | Previously a stateful apparatus blocker | The old failure is valid for that driver/hash, but two predicates have changed. The remaining sequential scorer/master cross-binding gate is real and still unproven. | **RE-SCOPED** to `INSTANCE(old landed driver/hash)` | Current ps135 owns the live sequential bank. No na6 probe; harvest only a terminal, parity-checked bank. |
| A18 | Killing projected-global row starts because the whole projected candidate is worse conflicts with GEN-2 row-local evidence: 95 selected moves came from projected starts; pass 1 also is not convergence (`ddm_ps135_pose_resolve_20260810.md`, GEN-2) | Whole-candidate start instance vs row-local basin family | Both observations are valid in their own units. Whole-candidate scoring cannot reject row-local starts; accepted-row counts cannot prove convergence. | **RE-SCOPED** to `INSTANCE(whole projected candidate)`; pass-1 non-convergence **UPHELD** | Folded into the active ps135 radius-2/multistart controller. Do not retry the whole projected candidate as a direct replacement. |
| S1 | QA39/TR1 carried-ξ token inter-prediction costs `+12,262 B` for warp context and `+165,871 B` for innovation context (`.omx/research/ddm_xi1_carried_xi_inter_race_20260729.md`) | `INSTANCE(TR1 chart/alphabet/SMEVR/intra-pair t_p)` | Valid. Exact lossless token reconstruction and charged SMEVR bytes answer its rate claim. | **UPHELD** | Folded on TR1. Finer charts and inter-pair vector prediction remain outside scope. |
| S2 | #881: post-hoc pose solve on an ep854 seg-only base misses the matched-control floor by about 46× (`.omx/research/ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md`) | `FORMULATION(post-hoc pose solve on ep854 seg-only base)` | Valid. Matched 74-pair scorer control measures pose in the claim's units. ps135 starts from a different pose-carrying LC2 base and uses joint/sequential exact descent, so no numeric transfer is valid. | **UPHELD** | Folded at stated scope. ps135 is not a reopen of #881; it satisfies a different precondition. |
| S3 | #918: explicit-token lossless coding is closed on TR1; lp135 reaches the same conclusion on F26/LC2 (`.omx/research/ddm_bo1_seg_base_objective_menu_order_20260803.md`, `.omx/research/ddm_na3_negative_audit_20260804.md`, `.omx/research/ddm_hp1_20260806/RECEIPT.md`, `ddm_lp135_lossless_pack_20260810.md`) | `FAMILY(tested explicit-token lossless coders on named objects)`; learned ≤10K static prior separately scoped | Valid. Every source measures exact payload bytes, but on different symbol objects. This is convergent evidence, not a same-cell theorem. | **UPHELD** | Folded for the named explicit token objects. Lossy maps and different probability objects remain open. |
| S4 | #941 PE3 explicit target substitution and sr1 implicit entropy context were previously described as convergent (`LC1_RECEIPT.md`, `ddm_sr1_implicit_edge_conditioning_20260811.md`) | Two different cells: distortion/Seg substitution vs entropy/rate conditioning | Each instrument is valid for its own claim, but the cells do not match. Neither measures joint distortion-side conditioning on the current vehicle. | **RE-SCOPED**: convergence only at the broad “small post-hoc additions lose” pattern, not at mechanism/cell level | JS1/#995 owns the open joint-conditioning test; no duplicate na6 probe. |

## S3 — carried-ξ cross-vehicle reconciliation

The two negatives are family-consistent but not family-closing.

- QA39/TR1 conditions an explicit token predictor on carried pose state. It loses exact serialized
  bytes by `12,262 B` (warp) or `165,871 B` (innovation) on a 24×32, L16 token chart.
- sr1 conditions F26 entropy probabilities on scalar pose sign/delta. It loses `43 B` on the full
  LC2/PR135 stream.

Both show that a small, post-hoc pose-derived context does not pay for itself in the tested token
probability object. They are different vehicles, lattices, alphabets, and context functions, so the
joint evidence does **not** close carried pose information, finer inter-pair vector prediction, or
joint renderer conditioning. The honest conclusion is convergent `FORMULATION` evidence across two
instances, not a universal `FAMILY` verdict.

## Re-conditioning against the new state

### ps135 descending solve

The descending solve changes two preconditions but reverses no standing verdict:

- It demonstrates that a pose-carrying LC2 base plus exact sequential descent is productive. That
  does not transfer to #881's post-hoc solve on an ep854 seg-only base.
- It escapes the exhausted ±1 singleton neighborhood through radius-2/multistart row-local moves.
  This narrows any prose that said “search is closed,” while preserving the exact singleton close.
- It clears q4 parity and cold relocation, narrowing the old Stage-C apparatus blocker. It does not
  erase the still-required scorer/master cross-binding or make a partial pass terminal.

### lp135 and cp135 rate close

The newer rate work reinforces #918 only for exact lossless coding of the named explicit objects.
TR1, F26, and CP135 are different streams; matching signs are convergent empirical evidence rather
than a transferable compression theorem. The CP135 composed floor also falsifies the old FD135
“unbanked residue” premise by exact ancestry: all `4,328 B` were already included. No standing
negative that assumed an open ≥1 KB FD135 residue remains eligible.

### Device-axis risk

ps135 pass acceptance uses an exact full-population **CPU** scorer, so sampling is excluded as the
source of uncertainty. Device transfer is not excluded. LC2 exact bytes exhibit a 6.6× pose gap
between contest CPU and CUDA, and the PR102 precedent recorded a roughly 5× pose-device gap. These
numbers do not predict ps135's sign or magnitude. They establish only that the terminal archive must
be replayed on both axes before it can move a pointer or support a cross-device mechanism claim.

## State-only exclusions from the scientific denominator

- “Do not score direct-v6 now” is scorer-lane custody, not a negative. The scorer step is queued.
- ps135 pass 3 in-progress state is not a receipt and supplies no verdict.
- The pinned Torch import timeout and cold-store relocation concern were apparatus observations;
  GEN-2 exact parity and relocation checks supersede them. They cannot be cited as scientific
  negatives.
- Local CPU or macOS-CPU rows are advisory unless explicitly labeled contest-CPU. Their lack of
  authority is a boundary, not a negative result.

## Prior-law check

The pre-registered prior predicted 2–4 re-scopes and 0–1 reopen across roughly 12–15 grouped rows.
Element-level expansion produced 22 rows and exactly four re-scopes, zero reopens. The larger
denominator comes from refusing to merge different claim units: entropy bytes, realized Seg,
receiver realizability, CPU wall time, and device transfer remain separate. The result does not
indict the arc's verdict discipline. The audit's main correction is scope hygiene, not reversal.

## RECALL EVIDENCE

Recall was run before adjudication with `tools/corpus_query.py` across all seven stores
(`research`, `equations`, `memory`, `dag`, `council`, `tasks`, `docs`) and with direct reads of the
pinned receipts, canonical equation listing, canonical research indexes, DAG FEED blocks, and task
status surfaces. Queries included:

- `F26 additive causal edge calibration`; `F26 scalar pose cross stream calibration`;
  `checkerboard reorder HPAC group order`; `CPC1 exact causal partition replacement`;
- `PE3 explicit label target substitution conditioning only`; `#941 PE3 conditioning only`;
- `PGQ1 rank five low rank pose gauge`; `PK3 frozen post hoc CPR1 gauge 64 bytes`;
  `PGQ1 non renderable envelope receiver`; `PZ3 exact coefficient residual`;
  `G91 PoseNet outputs SE3 physical warp`;
- `F26 ANS RC64 same state`; `CAP1 CPR1 representation mismatch FD135 residue`;
  `PR135 SMEVR LOTTO coder race`; `rate coding closed explicit token`;
- `LC2 native entropy CPU cure HPAC cache`; `LC2 CPU CUDA device delta pose`;
- `PR135 singleton int12 PR133 integer code transfer compensation`;
  `#881 pose carrying base ep854`; `QA39 carried xi token inter prediction`.

Beyond the charter seeds, recall found three decision-changing surfaces:

1. QA39's exact receipt showed the same broad post-hoc context pattern as sr1 but a materially
   different token object, preventing a family-level merge.
2. LC1's PE3 receipt showed that #941 measured explicit target substitution in realized Seg units,
   whereas sr1 measured entropy rate. This caused re-scope S4.
3. GEN-2's exact q4 parity, cold relocation, 95 projected-start selections, and radius-2 progress
   narrowed two ps135 dead-end statements without reopening the rejected whole candidates.

No additional source in the searched canonical equations, DAG, council, task, or docs surfaces
contradicted the pinned full-population/sample-valid negatives. The live cn4 consolidation memo was
read in the shared worktree for consumer names and fire-order alignment; na6 made no lane-registry or
canonical-equation writes.

## Boundaries

- No scorer, renderer, encoder, materializer, Modal job, or public-PR mutation was run.
- No score in this memo is promoted from macOS-CPU advisory to contest authority.
- No partial ps135 pass is treated as a candidate or convergence receipt.
- No cross-vehicle number is transferred. Cross-vehicle matches are labeled convergent patterns.
- No `FAMILY` statement exceeds the tested symbol object and mechanism family.
- The exact frontier pointer did not move in this scorer-free audit. This is an audit result, not
  goal progress.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN exact-replay owner; consumer store: terminal ps135
  archive receipt plus the canonical exact-eval/pointer store; fire trigger: ps135 emits a terminal,
  byte-closed, parity-checked candidate and the scorer lane is free. Replay the identical archive on
  contest-CPU and contest-CUDA before any pointer decision.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1/#995; consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/`; fire trigger: the
  terminal ps135 base and the registered stage-0 edge decomposition are available. Test joint
  distortion-side conditioning; do not retry explicit PE3 target substitution or scalar-only entropy
  context.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: pz4r scorer successor; consumer store: the retained
  direct-v6 receiver/archive receipt named by `.omx/research/ddm_pz4r_pgq1_receiver_20260811.md`;
  fire trigger: ps135 releases the full-n600 scorer lane. Score the exact direct-v6 bytes; do not use
  the non-renderable `168,005 B` envelope as the candidate.

## LIVE-HYPOTHESES

- Joint distortion-side conditioning may pay where both explicit PE3 substitution and small entropy
  contexts failed, because it changes the rendered evaluator cells during optimization instead of
  appending a post-hoc rate context. JS1/#995 is the existing owner.
- Radius-2/multistart row-local pose search may keep descending after ±1 singleton exhaustion,
  because GEN-2 pass 1 accepted 597 moves, pass 2 accepted 517, and 95 pass-1 selections came from
  projected starts. Only a terminal receipt can establish the endpoint.
- The pz4r direct-v6 receiver may still be a useful scored rate trade, because it is byte-closed and
  saves `4,089 B`; its pose/Seg effect, not its non-renderable preproof envelope, is the unresolved
  quantity.
- Native lowering or parallel restructuring of HPAC probability generation may cure CPU runtime,
  because measured entropy decode is only `1.1–3.1 s` and more than 99.5% of the wall is elsewhere.

## DEAD-ENDS

- Small post-hoc pose/edge entropy contexts on the pinned F26 formulation: held-out rate and exact
  serialized bytes do not pay for their tables.
- QA39's tested carried-ξ warp and innovation contexts on TR1: both enlarge the exact lossless
  payload; this does not close finer charts or inter-pair vector prediction.
- Explicit PE3 target substitution: all 32 stratified frames worsen after realized-through-R replay
  and the side information costs `74,408 B`.
- Frozen PK3 post-hoc gauges and PGQ1 exact-residual repair: the first saves only `64 B`; the second
  adds `4,316 B`.
- PGQ1's `168,005 B` envelope as a rendered candidate: it is not receiver-realizable; direct-v6 is
  the honest byte-closed object.
- F26 same-state ANS, CAP1 repetition, CP135 SMEVR/LOTTO/supermask, and the claimed unbanked FD135
  residue: exact bytes or ancestry close each named form.
- Native entropy-decode replacement and the tested cached HPAC plan as complete CPU cures: measured
  wall attribution or timing rejects them; probability generation remains the target.
- PR135 exact ±1 singleton search, direct PR133 integer-code copy, #740 direct CPR1 reuse, #460 n1
  precision transfer, uncompensated mixed precision, and the whole projected-global replacement:
  each is closed at its stated instance/formulation scope. Radius-2 row-local search is distinct.
