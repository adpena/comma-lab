# ddm_ah2 — full finished-arm harvest against the 0.162 bar

**Date:** 2026-08-10

**Scope:** the 41 clean-finished, live-marked arms reported by
`tools/codex_arm_queue.py status`

**Status:** COMPLETE — 41/41 adjudicated, 0 UNKNOWN, no scorer or payload run

## Outcome first

The exact pointer did **not** move. The own-custody anchor remains
**S = 0.16959899569230852 @ 187,226 B**
`[contest-CUDA T4, locked upstream venv, n600]`, archive SHA-256
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.
The competitive target is the official-display PR #135 row at **0.162**; its
extra displayed precision is unavailable, so all comparisons below use the
displayed value and say so.

The sweep produced five route-worthy joins and one representation candidate
that re-ranks below 0.162 only under an explicit no-receiver toy bracket. It
did **not** produce a new exact candidate below 0.162:

1. **Lossless composition:** VP1's measured `-903 B` split-model result and
   HP3's receiver-closed `-8 B` step2 result can be composed on LC2 only by
   materializing a new archive. The byte-additive projection is 186,315 B and
   S = 0.168992398186014 if decoded output remains LC2-identical. It is still
   +0.006992398 above the displayed bar and is not a score or built archive.
2. **Pose representation:** PZ2's 2,860 B direct DALI-target packet projects to
   S = 0.1586999 under its additive-error toy bracket and 0.1611043 under its
   worst-aligned toy bracket. It is the only harvested representation that
   numerically re-ranks below 0.162, but it has no frame receiver. PZ3 proves
   that exact-residual realization on the frozen carrier is larger than the
   base; only a jointly trained target-conditioned receiver remains live.
3. **Current-object distortion routing:** SD2R provides the retained,
   resumable decomposition apparatus and LT1 supplies five conditional
   force ports. They must be rebound to the current best object after PR135
   intake rather than fired as an obsolete PR130 characterization job.
4. **Trainable-port readiness:** FX3 and PQ1 close the semantic EMA/resume and
   row-local pose-optimizer implementation gaps. RR1/RR4 still require typed
   provenance and real current-object execution before a training claim.
5. **Negative/corpus routing:** NB2 leaves three formulation-scoped retests,
   and VH2R leaves the `v` lineage as the largest unharvested vehicle
   partition. Both have explicit, lower-priority fire triggers below.

The prior-law prediction is therefore supported only at the routing/projection
level: at least three route-worthy findings were found, and one no-receiver
representation bracket is below 0.162. It is **not** supported by an exact
score row; the pointer is unchanged.

## Candidate pricing against 0.162

The rate dual is `25 / 37,545,489 = 6.658589531221713e-7 S/B`. Except for the
PZ2 rows copied from its explicitly labeled toy bracket, each projection below
holds LC2's measured distortion fixed. That assumption is false for the lossy
CP2 rows and is shown only to price the rate headroom before the later
distortion evidence is applied.

| candidate | real/projected archive | projected S | gap to displayed 0.162 | adjudication |
|---|---:|---:|---:|---|
| LC2 exact anchor | 187,226 B | 0.169598995692309 | +0.007598995692309 | measured exact anchor; 11,412.32 rate-equivalent bytes above the displayed bar |
| LC2 + split model | 186,323 B projected | 0.168997725057639 | +0.006997725057639 | `-903 B` measured section result, but public combined archive unbuilt; QUEUED for real composition |
| LC2 + split model + HP3 | 186,315 B projected | 0.168992398186014 | +0.006992398186014 | byte-additive projection only; 10,501.32 further rate-equivalent bytes still owed |
| CP2 low-rank r32 + temporal | 182,364 B real | 0.166361589462229 if LC2 distortion held | +0.004361589462229 | FOLDED: SM4 proves the exact decoded state is the already-refuted n600 advisory row, S about 7.4924 |
| CP2 joint VQ32 + temporal | 183,992 B real | 0.167445607837911 if LC2 distortion held | +0.005445607837911 | FOLDED post-hoc instance: SV3 measured odd-frame MAE 6.255 and 94.03% changed; trained VQ remains conditional |
| CP2 mixed q3/q4 + temporal | 187,788 B real | 0.169973208423963 if LC2 distortion held | +0.007973208423963 | FOLDED as the next exact spend after the bar move; prior semantic leg is favorable but Pose is unmeasured and the row does not attack the gap directly |
| PZ2 direct p092 target packet | 170,528 B hypothetical | 0.1586999 additive-error toy bracket; 0.1611043 worst-aligned toy bracket | -0.0033001 / -0.0008957 | CANDIDATE below displayed bar only after a real counted receiver; exact control is blocked before scorer |

HP3's eight-byte survivor is exact-output and receiver-closed, but its
standalone evaluation is dominated by LC2. It is retained only as a component
of the real lossless composition. The `-903 B` split-model result is likewise
not promoted as an archive: VP1 explicitly records that the final unchanged-q4
public archive was not separately banked.

## Fired, folded, and queued routes

Every nonterminal finding below has exactly one owner, consumer, and trigger.
Nothing is left as “noted” or UNKNOWN.

### Q1 — lossless LC2 composition — QUEUED-WITH-FIRE-ORDER 1

- **Owner:** MAIN PR135/PR130 lossless-pack composer.
- **Consumer:** `/Volumes/VertigoDataTier/pact/ddm_ah2_lossless_compose_20260810/`
  plus the exact-claims store.
- **Fire trigger:** PR135 depth intake is terminal and shows whether split
  models and HP3-style frame-embedding requantization are already absorbed.
  If not, materialize LC2 + split model first, then add HP3 only through a
  fresh real recode. Retain every archive and decoded payload; require exact
  output equality and a real archive below 187,226 B. Exact contest-CUDA fires
  only after #1008 bare-venv bootstrap passes and the lane is claimed.

### Q2 — joint target-conditioned pose receiver — QUEUED-WITH-FIRE-ORDER 2

- **Owner:** `ddm_pz4_joint_target_conditioned_receiver` successor.
- **Consumer:** `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/`.
- **Fire trigger:** bind the retained PZ2 p092 packet to the best available
  PR135/LC2-compatible base, land a deterministic resumable joint trainer with
  per-stage checkpoints, and pre-register a full-archive ceiling below that
  base. Reject before scoring unless the receiver removes the frozen
  coefficient residual and the complete archive remains below the incumbent.

### Q3 — current-object segmentation decomposition — QUEUED-WITH-FIRE-ORDER 3

- **Owner:** MAIN sole-scorer owner.
- **Consumer:** `/Volumes/APDataStore/pact/ddm_sd2_20260810/` and LT1's
  `RESULTS.md` force-selection surface.
- **Fire trigger:** PR135 intake supplies a current-object archive/runtime or
  explicitly concludes PR130 remains the editable base; the scorer lane is
  exclusively claimed; the SD2R retention/admission checks pass. Measure the
  current object, then admit only the LT1 force whose directed cell is material.

### Q4 — current-object resumable training port — QUEUED-WITH-FIRE-ORDER 4

- **Owner:** task #995 / current PR135-derived semantic-and-pose training owner.
- **Consumer:** the typed checkpoint ledger and a new SSD stage-checkpoint root.
- **Fire trigger:** Q3 identifies a material trainable cell and PR135 intake
  confirms that the section is not already optimized away. Reuse FX3 EMA and
  atomic resume, PQ1 row-local optimizer semantics, and RR1/RR4 provenance
  guards; no launch before real device availability and per-stage retention.

### Q5 — representative negative retests — QUEUED-WITH-FIRE-ORDER 5

- **Owner:** the existing NB2/NA5 named retest owners.
- **Consumer:** `.omx/state/probe_outcomes.jsonl` and the prepared representative
  sample stores.
- **Fire trigger:** a current vehicle actively requests Yousfi native-grid,
  ego-hood, YOPO reuse, or a post-hoc pose formulation and the source-faithful
  positive control reproduces. Old prefix evidence alone cannot fire a verdict.

### Q6 — remaining vehicle lineage harvest — QUEUED-WITH-FIRE-ORDER 6

- **Owner:** next MAIN vehicle-harvest arm.
- **Consumer:** VH2R's canonical routing ledger and coverage reader.
- **Fire trigger:** Q1/Q2/PR135 intake are not waiting on MAIN and the unrelated
  shared-ledger changes have custody. Drain the 121-root `v` partition next;
  do not revive its old n64 entropy number as PR130 evidence.

### Q7 — exact-runtime submission gate — QUEUED-WITH-FIRE-ORDER 7

- **Owner:** task #1008 / exact-runtime custodian.
- **Consumer:** the bare-venv bootstrap receipt and the next exact candidate's
  runtime manifest.
- **Fire trigger:** before any submission or new exact candidate dispatch.
  Prove Linux bootstrap and evaluator three-argument behavior on the exact
  runtime; AX2's mixed-axis wrapper must not be used for a CUDA claim.

## Per-arm disposition ledger — 41/41

| arm | one disposition | routed finding or terminal reason |
|---|---|---|
| `ddm_fx2` | LANDED-CONSUMED | Three-argument evaluator adapter and raw identity were consumed by LC2; remaining bootstrap proof is Q7. |
| `ddm_pk4` | EMPTY-honest | 432-candidate declared gauge bank saved only 64 B versus the 2,000 B trigger; QAT fire FOLDED. |
| `ddm_rr4` | FINDING→ROUTED | Silent lift drift, native-sparse selection, and non-atomic carrier state route to Q4 before another training claim. |
| `ddm_sr1` | LANDED-CONSUMED | Variable-length SD1M receiver and full raw-identity proof were consumed by CP2/SV3; duplicate parser/scorer actions FOLDED. |
| `ddm_rr2` | LANDED-CONSUMED | FX3 consumed the EMA/resume and typed-reader defects; the corrected 20.6486-GFLOP denominator remains the throughput authority. |
| `ddm_rr1` | FINDING→ROUTED | FX4 consumed the DALI-cache confound; remaining source/selected-checkpoint causal verification routes to Q4. |
| `ddm_ax2` | FINDING→ROUTED | DALI target selection was consumed by LC2; the mislabeled CUDA/AV wrapper routes to Q7 and is forbidden for authority. |
| `ddm_cb2` | LANDED-CONSUMED | Its 191,044 B head item became HP3, and ANS/temporal became LC2; old 0.172-bar ordering is superseded. |
| `ddm_cp2` | CANDIDATE→PRICED vs 0.162 | Three real archives priced above; low-rank and VQ post-hoc rows FOLDED by SM4/SV3, mixed immediate exact spend FOLDED after the bar move. |
| `ddm_dg1` | LANDED-CONSUMED | Predicate class cure and guard landed; no score-route follow-on remains. |
| `ddm_dt1` | LANDED-CONSUMED | Retained ANS payload was consumed byte-identically by AI1/TM1 and the LC2 exact row. |
| `ddm_dv1` | LANDED-CONSUMED | FX2 consumed the runtime repair; LC2 later measured the current CPU instance budget-infeasible, so duplicate CPU/AV dispatch FOLDED. |
| `ddm_fx1` | LANDED-CONSUMED | Constriction dependency closure was consumed by LC2; Q7 owns the remaining bare-host proof. |
| `ddm_fx3` | FINDING→ROUTED | EMA, typed schema, and crash-resume implementation route to Q4; no training or score was claimed. |
| `ddm_fx4` | LANDED-CONSUMED | DALI cache became the canonical PR130/LC2 axis; another generic AV-vs-DALI job is FOLDED. |
| `ddm_fx5` | LANDED-CONSUMED | FX5B closed stale claims and LC2 later exercised locked Linux dependencies; the final bare-venv gate is Q7. |
| `ddm_fx5b` | FINDING→ROUTED | Provider/budget-era blocker is superseded; exact Linux bootstrap obligation routes to Q7. |
| `ddm_hm1` | FINDING→ROUTED | Post-hoc coordinate-5 deletion is rate-negative; trained capacity reallocation remains Q4 only after a current-object trigger. |
| `ddm_hp3` | FINDING→ROUTED | Receiver-closed `-8 B` exact-output lever routes only into Q1; standalone exact evaluation FOLDED as dominated by LC2. |
| `ddm_hr1` | LANDED-CONSUMED | 221/221 follow-ons were already routed and the NEXT_IF_RESUMED generator cured; no re-routing duplicate was created. |
| `ddm_lt1` | FINDING→ROUTED | Five PR130 forces are conditional on measured directed errors and route to Q3/Q4; two exact-mechanism ports remain dead on unchanged CPR1. |
| `ddm_mp2` | LANDED-CONSUMED | Managed-sandbox Metal absence was superseded by PQ1's port and RR4's real native-sparse receipt; retrying this sandbox is FOLDED. |
| `ddm_na5` | FINDING→ROUTED | 0/4 formulations were measured because the scorer stop fired; representative source-faithful reruns route to Q5. |
| `ddm_nb1` | LANDED-CONSUMED | NB2 adjudicated the 138 unreached bodies and NA5 prepared the representative pose samples. |
| `ddm_nb2` | FINDING→ROUTED | Yousfi, ego-hood, and YOPO reopens route to Q5 with their existing task identities and fire order. |
| `ddm_pk2` | LANDED-CONSUMED | PK4 executed its sole gauge-QAT trigger and failed it; frozen-carrier pose recodes/drops remain closed. |
| `ddm_pp2` | LANDED-CONSUMED | PQ1 built the row-local adapter and RR4 located the native-sparse pass; whole-pose claims still route through Q4. |
| `ddm_pq1` | FINDING→ROUTED | Exact CPU optimizer equivalence is built; real current-object Metal execution routes to Q4. |
| `ddm_pz2` | CANDIDATE→PRICED vs 0.162 | The 2,860 B packet is below the displayed bar only in explicit toy brackets; Q2 owns the missing counted frame receiver and later exact control. |
| `ddm_pz3` | FINDING→ROUTED | Exact-residual frozen-carrier realization is +3,068 B and FOLDED; the formulation-scoped wall routes only the joint-trained receiver to Q2. |
| `ddm_rc2` | LANDED-CONSUMED | RC2R executed the storage-restored adaptive/LDPC race; the first blocked receipt has no separate follow-on. |
| `ddm_rc2r` | EMPTY-honest | PPMd loses on every unchanged section and LDPC loses by at least 540,909 B; unchanged-object coder families are FOLDED. |
| `ddm_sd2` | LANDED-CONSUMED | SD2R repaired Range/ANS resume capability and preserved the runnable retained-decomposition apparatus. |
| `ddm_sd2r` | FINDING→ROUTED | The apparatus is runnable but old-object firing is withheld; current-object rebind and sole-scorer fire route to Q3. |
| `ddm_sg2` | FINDING→ROUTED | Stage-08 is already shipped and mixed q3/q4 cannot be promoted from its semantic leg; exact edge decomposition routes to Q3. |
| `ddm_sm3` | LANDED-CONSUMED | CP2 receiver-closed its top modes; SM4 and SV3 supplied the missing cheap screens, so the original six-row scorer queue is superseded. |
| `ddm_sm4` | EMPTY-honest | Uniform low-rank precision/centering rescue is formulation-dead at the matched budget; no duplicate scorer row. |
| `ddm_sv3` | FINDING→ROUTED | Post-hoc VQ and low-rank are FOLDED; mixed immediate exact spend is FOLDED against the new bar; trained VQ may enter Q4 only after current-object evidence. |
| `ddm_vh2` | LANDED-CONSUMED | Permission-blocked first pass was consumed by VH2R after SSD access was restored. |
| `ddm_vh2r` | FINDING→ROUTED | Forty-eight rows are owned; the 121-root `v` partition routes to Q6, while the old n64 entropy result stays non-transferable. |
| `ddm_vp1` | FINDING→ROUTED | Its ANS/temporal rows were consumed by LC2; the unbuilt `-903 B` split-model lever and HP3 join route to Q1. |

Disposition counts: **19 LANDED-CONSUMED, 17 FINDING→ROUTED,
2 CANDIDATE→PRICED, 3 EMPTY-honest = 41 total**. Every candidate or finding
that survives has a Q1–Q7 fire order; all other proposed follow-ons are folded.

## RECALL EVIDENCE

The harvest did not rely on charter seeds alone.

- Queue population: latest-field merge of
  `.omx/state/codex_arm_queue.jsonl`, checked against all 41 `.done` receipts
  and all 41 nonempty `.last.txt` captures. Query surface:
  `tools/codex_arm_queue.py status`, `.done`, `.last.txt`, and the persisted
  `arm_final_messages/` copies.
- Per arm: final message, `git log --all --oneline --grep=<arm>`, then
  file-history lookup for the linked findings memo. Current-main memo custody:
  37 arm outputs tracked, HM1/VH2R/VP1 present but untracked with their own
  fallback/blocked receipts, and VH2 correctly wrote no memo after its mandated
  SSD stop. Sister artifacts were not edited.
- Full-corpus content searches covered `182364`, `pointwise_lowrank_r32`,
  `split model`, `-903`, `PR135`, `0.162`, `ddm_cp2`, `ddm_vp1`, and task
  `#1006` across research indexes, the sub-0.15 DAG, live hot state, task/P0
  ledgers, and the named arm memos.
- The 429-entry canonical-equations registry was enumerated with
  `tools/list_canonical_equations.py --json`. The load-bearing law is the
  measured byte-only rate dual; projections remain non-score claims until
  exact receiver/evaluator closure.
- Pointer truth was re-read from
  `.omx/state/canonical_frontier_pointer.json`: PR135 0.162 is an
  official-display external target; LC2 0.16959899569230852 is the local
  contest-CUDA anchor.

Findings beyond the charter's head seeds changed the plan materially:

1. SM4's later exact-state identity closes CP2 low-rank as the next scorer row;
   the charter's “exact-axis control owed” is superseded for that instance by
   the retained n600 advisory catastrophe and no-rerun law.
2. SV3 withdraws post-hoc joint VQ before scorer, leaving only trained VQ.
3. PZ2 supplies the only below-0.162 numerical representation bracket, while
   PZ3 identifies the exact realization wall and the only honest successor.
4. LC2 already consumed ANS and temporal reversion, so VP1's live lossless
   remainder is split-model composition, not another coder race.
5. The bar move makes the old mixed-q3/q4 exact queue a poor immediate spend;
   same-object PR135 intake and a gap-crossing construction now precede it.

## Verification and boundaries

- **Measured in this arm:** queue denominator 41/41; receipt/final-message
  presence; current Git custody; arithmetic projections recomputed from the
  exact rate formula; current pointer and official-display bar.
- **Consumed measurements:** the arm-owned byte counts, hashes, parse-back,
  retained RAW, advisory rows, and exact LC2 row cited with their original axes.
- **Not measured:** no scorer, evaluator, decoder, trainer, Modal job, public
  archive intake, payload materialization, or paid compute ran in AH2.
- No score claim is made for CP2, HP3, VP1 split streams, PZ2, or PZ3.
- The exact pointer is unchanged and remains above both the 0.162 displayed bar
  and the sub-0.15 goal. This harvest is a routing artifact, not goal progress.
- The staged index and all unrelated dirty work were preserved. Protected
  sister files and append-only arm artifacts were not edited.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER 1** — owner: MAIN PR135/PR130 lossless-pack composer;
  consumer: `/Volumes/VertigoDataTier/pact/ddm_ah2_lossless_compose_20260810/`;
  fire trigger: PR135 depth intake is terminal and confirms split-model/HP3 are
  not already absorbed; materialize and retain the real LC2 composition.
- **QUEUED-WITH-FIRE-ORDER 2** — owner: `ddm_pz4_joint_target_conditioned_receiver`;
  consumer: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/`;
  fire trigger: a current-base, resumable, per-stage-checkpointed receiver can
  consume PZ2 p092 and remove the frozen residual under a full-archive ceiling.
- **QUEUED-WITH-FIRE-ORDER 3** — owner: MAIN sole-scorer owner; consumer:
  `/Volumes/APDataStore/pact/ddm_sd2_20260810/`; fire trigger: PR135 intake
  selects the current object and the exclusive scorer/storage gates pass.
- **QUEUED-WITH-FIRE-ORDER 4** — owner: task #995 current-object trainer;
  consumer: typed checkpoint ledger plus a new SSD stage root; fire trigger:
  Q3 identifies a material trainable cell not already absorbed by PR135.
- **QUEUED-WITH-FIRE-ORDER 5** — owner: existing NB2/NA5 retest owners;
  consumer: `.omx/state/probe_outcomes.jsonl`; fire trigger: a current vehicle
  requests the formulation and its source-faithful positive control reproduces.
- **QUEUED-WITH-FIRE-ORDER 6** — owner: next MAIN vehicle-harvest arm; consumer:
  VH2R canonical ledger; fire trigger: Q1/Q2/PR135 intake are not waiting on
  MAIN and shared-ledger custody is clean; drain the 121-root `v` partition.
- **QUEUED-WITH-FIRE-ORDER 7** — owner: task #1008 exact-runtime custodian;
  consumer: bare-venv bootstrap receipt; fire trigger: before any new exact
  dispatch or submission.

## LIVE-HYPOTHESES

- A real LC2 + split-model archive should retain most of the measured 903-byte
  gain because the source measurement used unchanged q4 model sections and
  explicit framing; final-container interaction is the remaining uncertainty.
- HP3 step2 may add a small extra win to that archive, but its `-8 B` is not
  safely additive with changed model/token compression and must be re-coded.
- A jointly trained target-conditioned pose receiver can still exploit PZ2's
  2.9 KB target packet because it may delete the 9.8 KB exact residual that
  made PZ3 lose; the frozen-carrier formulation could not change the frames.
- Rebinding SD2/LT1 to PR135 may expose a concentrated directed semantic cell
  that is absent from the polished public row; this is plausible but wholly
  unmeasured until PR135 archive/runtime intake exists.
- Heterogeneous or residualized semantic representation training may outperform
  post-hoc low-rank/VQ, because SM4/SV3 close only the post-hoc formulations.

## DEAD-ENDS

- CP2 low-rank r32 exact-state rescoring: closed by SM4's state identity to the
  retained n600 advisory catastrophe.
- Post-hoc joint VQ32 promotion: closed by SV3's broad raw damage; only trained
  representation-aware VQ remains.
- Selecting CP2/SM3 candidates by bytes, weight MSE, singular energy, or prior
  semantic-only results: closed; both score components are mandatory.
- Exact-residual PZ2 realization on the frozen PR130 carrier: closed by PZ3's
  3,068-byte regression across all twelve cells.
- PPMd recoding of unchanged sections and LDPC/BP coding of the current HPAC
  hit field: closed by RC2R's retained real payload race.
- The declared PK4 gauge bank and frozen-carrier PK2 recodes/dimension drops:
  closed at their stated instance/formulation scopes.
- Post-hoc HM1 coordinate-5 deletion: closed because token growth overpays the
  model saving.
- Re-running generic AV-versus-DALI attribution or treating the target-to-target
  difference as candidate score: closed by FX4/AX2.
- Re-running the old mixed-q3/q4 row merely to beat the obsolete 0.172 bar:
  closed after PR135 moved the target to 0.162.
- Repeating the old n64 entropy result or transferring v7–v19/TR1 numbers onto
  PR135/PR130: closed as wrong object, receiver, and authority surface.
