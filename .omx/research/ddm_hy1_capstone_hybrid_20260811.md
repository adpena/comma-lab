# ddm_hy1 — capstone hybrid: C1 solve × PR135 carriage × joint realization

**Date:** 2026-08-11  
**Status:** scorer-free head probes complete; composed build and scorer legs queued  
**Axis:** `[macOS-CPU scorer-free real-coder n600]` unless a row states otherwise  
**Score claim:** false  
**Pointer movement:** none

The effective frontier remains **cp135 `S = 0.16195513827824176 @ 186,252 B`
`[contest-CUDA T4, n600]`**, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
The own-vehicle authority remains **LC2 `S = 0.16959899569230852 @ 187,226 B`
`[contest-CUDA T4, adjudicated, n600]`**, archive SHA-256
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.

## Outcome

The solved-partition carriage direction passes both preregistered scorer-free gates and is the
campaign head. The exact C1 batch-16 solved semantic plane costs **114,717 B** under the frozen
F26 HPAC model plus native RC64, only **+11 B** versus F26's shipped 114,706-byte stream. An
independent causal decoder reproduced all **117,964,800/117,964,800** tokens exactly. The dense
PR135 grammar can syntactically express all **27,351/27,351** token changes between the retained
PR135 shipped plane and the C1 solved plane.

That does **not** establish a candidate score. The learned renderer, uint8/resize round trip, and
SegNet may fail to realize some or all of those token changes. Moreover, cp135 carries a different
HP3 probability object and a 115,231-byte token stream. The +11-byte F26 result is not additive to
cp135; a whole-container rebuild and joint scorer replay are mandatory.

### Probe results

| Probe | Result | Type | Disposition |
|---|---:|---|---|
| C1 exact event count | **17,926 / 117,964,800**, `d_seg = 0.00015196058485243054` | MEASURED, retained C1 batch-16 event bank | Corrects the charter's rounded shorthand of 17,927. |
| Frozen F26 HPAC + RC64 | **114,717 B**, SHA `9def0a4b...936e8` | MEASURED, real coder, n600 | PASS: +11 B, or +0.00959%, versus 114,706 B. |
| Ideal codelength | **917,731.379709 bits = 114,716.422464 B** | MEASURED from frozen causal probabilities | Native coder overhead is 0.577536 B. |
| Independent causal decode | **117,964,800 tokens, exact equality** | MEASURED | Parser/coder closure passes. |
| Brotli q11 control | **429,383 B**, +314,677 B | MEASURED, deterministic repeat | DEAD at INSTANCE scope as the generic dense-token control. |
| Token-grid representability | **27,351 / 27,351 = 100%** | MEASURED syntactic grammar fact | PASS; renderer/R/SegNet survival remains unmeasured. |
| C1 target versus PR135 shipped | **20,748 pixels** | MEASURED | Reference-surface fact, not the solved-versus-shipped numerator. |
| C1 solved versus target | **17,926 pixels** | MEASURED | The exact C1 batch-16 error numerator. |
| C1 solved versus PR135 shipped | **27,351 pixels** | MEASURED | The representability numerator. |

The preregistered prediction of within ±10% of 114,706 B passes. The >+8,000 B rate falsifier
does not fire; it is missed by 7,989 B. The ≥50% representability gate passes, and the <20%
falsifier does not fire.

### Reference and authority correction

The authoritative event bank for this probe is
`.omx/research/original_taskspace_inverse_witness_codec_20260725/c1_live_target_debt_n600_batch16.json`,
SHA-256 `0db8e47a...fbd3`, and it contains 17,926 events. The associated exact replay receipt is
`/Volumes/VertigoDataTier/pact/c1_batch16_exact_replay_20260726/11_batch_replay_receipt.json`,
SHA-256 `2d117579...a49cf`. Older surfaces reported 17,927 under a different reference/batch surface;
that number is not used here.

C1 is an inverse-solved RGB object measured on its retained CPU-Torch advisory reference. Its
semantic field is encoder-side guidance until the field is emitted through the current public
receiver and rescored. The known MLX-versus-Torch SegNet authority disagreement is up to three
argmax pixels per frame, approximately `1.5e-5` of one 384×512 frame, concentrated at tiny-margin
separatrices. Therefore MLX/MPS cannot promote the C1 number, and this memo does not transfer it
to cp135.

## Wire concentration

The allocation below is exact ideal codelength under the candidate's frozen F26 probabilities.
RC64 is one adaptive stream, so literal emitted bytes are not separable by pair, patch, or HPAC
group; the allocation is a diagnostic, not a section-by-section byte claim.

| Surface | MEASURED result |
|---|---:|
| C1 event density | 0.0151961% of pixels |
| Ideal-wire share at C1 event sites | 3.42544% |
| C1 event wire enrichment | 225.416× relative to pixel density |
| All 27,351 solved-vs-shipped sites' ideal-wire share | 4.88890% |
| All-disagreement wire enrichment | 210.858× |
| Pair event mass versus pair bits | Pearson 0.39077; Spearman 0.42549 |
| 64×64 patch event mass versus patch bits | Pearson 0.52825; Spearman 0.65468 |
| 190-group event mass versus group bits | Pearson 0.86425; Spearman 0.83125 |
| Top 1% event-ranked pair-patches | 14.604% events; 3.868% bits |
| Top 5% event-ranked pair-patches | 41.772% events; 17.195% bits |
| Top 10% event-ranked pair-patches | 62.825% events; 31.456% bits |

The concentration is real, but it does not justify building a sparse representation first. The
full dense solved plane already fits the F26 wire at +11 B. Sparse generator-plus-token work is
a fallback if renderer survival proves localized, not a rate-motivated prerequisite.

## Priced waterfall from cp135

The following arithmetic is deliberately non-authoritative. It combines separately measured
quantities only to set realization gates; no row is a candidate score.

- MEASURED: cp135 gap to 0.15 is `0.01195513827824176 S`.
- MEASURED: F26 shipped `d_seg = 0.00029639352578669786`; C1 batch-16 solved
  `d_seg = 0.00015196058485243054` on its advisory reference.
- DERIVED: full C1 Seg transport is worth at most
  `100 × (0.00015196058485243054 - 0.00029639352578669786) = -0.014443294093426732 S`.
- DERIVED: the measured +11 B would cost `+0.000007324448484343885 S` if it transferred
  unchanged into cp135. It does not transfer without a whole-container rebuild.
- DERIVED: sub-0.15 requires **82.8236457%** of the full C1 Seg gain at that +11 B proxy.

| Realized fraction of C1 Seg gain | DERIVED score proxy from cp135 plus +11 B |
|---:|---:|
| 0% | 0.1619624627 |
| 25% | 0.1583516392 |
| 50% | 0.1547408157 |
| 75% | 0.1511299922 |
| 82.8236457% | 0.1500000000 |
| 100% | 0.1475191686 |

The prior-law claim "realized Seg reach at least -0.004 S" remains CONJECTURE until the scorer
leg. At +11 B that gain would improve cp135, but it would not reach sub-0.15. The governing fire
gate is stricter: target at least 82.824% of the C1 gain, then price the actual whole archive.

PZ4R's real receiver archive is 183,137 B, 4,089 B smaller than LC2. That isolated byte delta is
DERIVED as `-0.0027226972593165587 S` at the contest denominator, but its distortion is unscored
and it is non-additive with ps135, cp135, and this head. The earlier 19,221-byte PGQ1 envelope is
nonrenderable and cannot be used as a pose price.

## The capstone vehicle

### A. Solved-partition carriage — campaign head

**Borrowed substrate:** PR135/F26 five-class dense token grammar, HPAC network and sparse
evaluator, fixed residual table, group ordering, RC64 recurrence, learned renderer, carrier, and
archive runtime.

**Ours-original:** exact C1 batch-16 solved semantic plane; retained solved-vs-shipped delta;
real F26 HPAC/RC64 measurement and independent decoder; event/wire concentration map; joint
realization gates and adaptation plan.

The build is a whole-object retarget, not a sidecar. First construct the terminal base's own
probability object and archive around the C1 token plane, then adapt or jointly solve the learned
renderer so the decoded RGB/R/SegNet path realizes the requested cells. Keep every probability
object, token stream, archive, decoded frame field, and scorer output. Admission requires exact
parse-back, deterministic repeat, full n600 CPU advisory scoring, and a joint S improvement.

The first scorer ladder measures survival at 25%, 50%, 75%, 82.824%, and 100% of the C1 Seg gain.
Those are analysis thresholds, not separate claims. If survival is below 82.824%, Stage B jointly
optimizes realization rather than pretending the semantic plane has transferred.

### B. Solver arsenal — current vehicle only

The solver consumes js1 Stage 0's per-pair × per-edge decomposition of the terminal shipping base.
It searches global `int12 × basis × FiLM` state jointly with the solved token head. Every proposal
is receiver-realized and priced as one object; per-axis gains are never added after the fact.

1. Use the ms3/ms4 margin-Fisher bundle only after its records join to actual current-receiver
   coordinates. It is a preconditioner, not an actuator and not an independent score claim.
2. Treat #580 as a resize-operator `range(A)`/kernel gauge canonicalizer. Its approximately 80.67%
   real-linear nullity is not a SegNet-null proof, and uint8/receiver closure is mandatory.
3. Do not import the old j11 one-dimensional Q3 split as a finished pose-null actuator. Its 1-D
   rays had zero useful nullity. A Q3-like branch may re-enter only as a higher-dimensional joint
   projection with retained Pose Jacobian and rank-4 Seg/margin Jacobian evidence.
4. Use ms2r tolerance homotopy only against receiver-realized cells and coder rows. Tighten cell,
   pose, and rate tolerances together, with stage-boundary checkpoints.
5. Import SQ2's true-convergence stopping and uncapped-step option, not its old solved-paint
   result. The n32 uncap100 instance had 0/32 convergence and `+0.7969` pose erosion; copying that
   route is closed.
6. Keep ps135's demonstrated native GN, wrong-sign GN, global starts, radius-2 neighborhoods, and
   exact population refresh. Their value is mechanism evidence on this vehicle, not authority for
   the unmeasured hybrid.

### C. Conditional level-set hybrid — fallback after survival localization

If the dense head misses the 82.824% survival gate, localize the failures with the retained
pair-patch and 190-group maps. The fallback represents smooth, high-margin strata with the
generator/level-set state and retains dense token authority on high-event patches. The receiver
may derive edge/level-set state as zero-counted conditioning only when it is generic computation;
video-derived learned state remains in the archive.

The surviving SR1 route is distortion-side conditioning: decoder-derived edge state may change
proposal generation and capacity allocation. SR1 closed the entropy-only family: causal edge
conditioning saved only 2 B and pose cross-stream conditioning cost 43 B because HPAC already
captures those contexts. Do not re-run probability-calibration variants or ship an explicit edge
mask. Compare the conditional model to an equal-parameter control and retain both payloads.

### D. Pose fusion — terminal-base composition only

ps135b remains the live scorer owner and is currently in pass 3. Its terminal coefficient state,
convergence curve, sensitivity map, and exact archive become the pose base. PZ4R may then race as
a byte-closed direct-v6 gauge representation; its -4,089 B rate fact is real, but its S is pending.
Use se(3)/Chasles and `e_p` rank-1 structure only as proposal coordinates unless a receiver-closed
coder proves a byte delta. Preserve PR133's quantize-then-compensate rule: compact the pose state,
then jointly compensate through the actual renderer and PoseNet. No post-hoc stored-target number
is transferred into this vehicle.

## Ranked build and fire plan

1. **Head whole-container build.** Rebuild the terminal ps135/cp135 probability object and archive
   around the exact C1 plane; independent-decode every token and retain a deterministic repeat.
2. **Stage 0 scorer receipt.** On the sole scorer lane, retain terminal-base and head decoded RGB,
   argmax, per-edge error, Pose outputs, archive bytes, and recomputed S.
3. **Joint realization.** If direct survival is below 82.824%, optimize semantic head plus
   `int12 × basis × FiLM` with the safeguards in Stage B. Save every stage and complete candidate.
4. **Conditional fallback.** Only if the failed survival is localized, use the measured
   pair-patch/group concentration to split dense tokens from level-set-conditioned smooth strata.
5. **Pose/rate fusion.** Rerank ps135 terminal against PZ4R/direct-v6 and Stage-C compensation as
   whole candidates. Never add their isolated deltas to the head.
6. **Authority replay.** A retained archive that strictly improves full-n600 local advisory S and
   respects the byte ceiling is eligible for one exact contest replay by MAIN.

## js1-RESEAL AMENDMENT BLOCK

**Status:** HY1 scorer-free gates PASS. Promote solved-partition carriage to the first js1
distortion head, but do not claim that C1 `d_seg` transfers through PR135.

**Inputs to bind:**

- C1 solved tokens:
  `/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/c1_solved_tokens_n600.u8`,
  117,964,800 B, SHA-256
  `2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5`.
- Frozen-F26 HPAC/RC64 stream:
  `/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/c1_solved_tokens_n600.f26_hpac.rc64`,
  114,717 B, SHA-256
  `9def0a4ba849757d473ba2a23cb0fd5370f2566355e5a5cfd398f847349636e8`.
- Independent decoded tokens: same semantic SHA-256, all 117,964,800 tokens exact.
- Wire allocation:
  `/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/hpac_wire_allocation_n600.npz`,
  SHA-256 `3d3f8d12...790f`.
- Exact C1 numerator correction: **17,926**, not 17,927.

**Resealed stage order:**

0. Bind the ps135 terminal archive and cp135 composition receipts. Produce and retain the shipping
   base's per-pair × per-edge Seg decomposition before proposing a correction.
1. Build the exact terminal-base probability object on the C1 semantic plane, whole-container
   recount it, independently decode it, and keep both archive repeats. The F26 +11 B row is a
   calibration only.
2. Score the terminal base and direct solved-token head through the same real receiver. Retain RGB,
   argmax fields, Pose outputs, per-edge rows, archive facts, and recomputed components.
3. If realized C1 Seg gain is below **82.8236457%**, run the joint realization solve over semantic
   head plus `int12 × basis × FiLM`. Use joined margin-Fisher preconditioning; constrain #580 only
   as `range(A)` geometry; permit Q3 only as a higher-dimensional measured-Jacobian projection;
   use tolerance homotopy and true-convergence uncapped GN without importing SQ2's failed result.
4. Jointly rerank implicit distortion conditioning, adaptive mixed precision with compensation,
   and PZ4R/direct-v6 pose fusion. Every row is a complete retained archive; no isolated delta is
   added to another row.
5. Admit only a byte-closed candidate with strict full-n600 local S improvement. MAIN alone may
   dispatch exact contest CPU/CUDA replay.

**Fresh prior:** at the measured +11 B F26 proxy, sub-0.15 requires 82.8236457% of the full C1
Seg gain. Recompute that gate from the terminal archive; do not reuse it if ps135 changes the base.

**Owner:** js1/#995 successor after ps135 terminal.  
**Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`, with HY1 head
subtree `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`.

## Retained artifacts and reproducibility

The probe implementation is `experiments/ddm_hy1_capstone_hybrid_probe.py`. It is resumable at
stage boundaries and every 25 RC64 frames. It retains every materialized stream, deterministic
controls, model inputs, compiler outputs, checkpoints, and independent decoded plane.

- Store: `/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/`
- Store footprint: 383,792,390 manifest-recorded bytes in 140 files; `du` reports 366 MiB.
- Final result: `HY1_PROBE_RESULT.json`, SHA-256
  `516752a489b68665a0ecf99a18d19096d807cc9c4b67b550f446cb265e7ac92e`.
- Tree manifest: `TREE_MANIFEST.json`, SHA-256
  `6ba81981d7a89357efa7bc647a2c35d96ee7f6011bde9ea630c7fa5eea6be9fa`.
- Solved token payload: 117,964,800 B, SHA-256
  `2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5`.
- RC64 payload: 114,717 B, SHA-256
  `9def0a4ba849757d473ba2a23cb0fd5370f2566355e5a5cfd398f847349636e8`.
- Brotli q11 payload and repeat: 429,383 B each, both SHA-256
  `2f6cf99620e82b9315ecdc18f32684d16e99ad041a960d033db27acba3aabe2f`.

No generated payload was discarded. No scorer, evaluator, Modal, network, GPU, or MPS job ran.
The live ps135 store was read only.

## RECALL EVIDENCE

The full seven-store corpus was queried with `tools/corpus_query.py` using:

- `C1 solved partition PR135 HPAC token carriage realization`
- `margin Fisher resize projector pose null tolerance homotopy uncapped GN joint solve`
- `hybrid level set dense tokens flip mass wire layout pose gauge Chasles`

The searched stores contained 8,365 research rows, 886 equation rows, 2,105 memory rows, 915 DAG
rows, 297 council rows, 531 task rows, and 96 docs rows. Direct sources included the C1 custody
store and receipt; fd135 `EVIDENCE_MANIFEST`; eh1; pi135; hb1/hb2; pp1; v14; od9; js1; na6/cn4;
ms4d; ms2r; j11/j11r; SQ2; PZ4P/PZ4R; SR1; SX1; SG1; TR1; QA75; and the canonical equations
registry.

Findings beyond the charter seeds changed the plan:

- TR1/QA75 say realization, not partition rate, is binding, so the direct scorer ladder precedes
  new representation work.
- SR1 closes small post-hoc entropy contexts and leaves only decoder-side distortion conditioning;
  the memo does not propose another HPAC-context sweep.
- SX1 supports precision by distance to the separatrix, while SG1 warns that the flip set is
  transient; the conditional fallback uses current receiver diagnostics, not a static dictionary.
- MS4D's metric bundle is a preconditioner without its own actuator; it is admitted only after
  joining to current receiver coordinates.
- #580's nullity is resize geometry, not SegNet/PoseNet invisibility; the stage list adds explicit
  uint8 and scorer closure.
- J11's one-dimensional split had no useful pose-null space; the amendment replaces it with a
  higher-dimensional measured-Jacobian proposal.
- SQ2 uncap100 did not converge on n32 and badly eroded pose; only its convergence-control idea is
  retained.
- PZ4R supplies a real 183,137-byte receiver candidate, while the 19,221-byte PGQ1 envelope is
  nonrenderable; the pose branch is gated on PZ4R scoring.

## Honesty and non-additivity boundaries

- Grammar representability is not renderer survival and is not a SegNet result.
- C1's advisory `d_seg` is not a cp135 component until current-receiver scorer replay.
- F26's +11 B is not cp135's token-stream delta; cp135 uses HP3 and a different stream.
- PZ4R's -4,089 B is not a score win until distortion is measured, and it cannot be added to ps135.
- Margin-Fisher, resize-null, pose-null, homotopy, and GN mechanisms are solver ingredients, not
  score units.
- The score proxies in this memo are gates only. Only `upstream/evaluate.py` on the exact retained
  archive can move the contest pointer.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: HY1/js1 whole-container builder. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`. Fire trigger: ps135 emits its terminal safe-run receipt and final archive; bind that base, adapt its probability object to the retained C1 plane, recount the complete archive, and prove independent token decode before requesting a scorer lane.**
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: js1/#995 scorer successor. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`. Fire trigger: ps135 is terminal, the sole scorer lane is freshly claimed, and the whole-container HY1 archive is retained; run Stage 0 plus the direct solved-head scorer ladder, keeping RGB, argmax, Pose, per-edge, archive, and score-component payloads.**
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: js1 joint-realization successor. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/joint_realization/`. Fire trigger: the direct head realizes less than the freshly recomputed sub-0.15 fraction; launch the checkpointed joint `int12 × basis × FiLM` solve with the amendment safeguards and retain every candidate.**
- **Disposition: CONDITIONAL-QUEUED. Owner: HY1 conditional-representation successor. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/conditional_levelset/`. Fire trigger: scorer-retained failures are localized enough that a dense-token/level-set split has a measured receiver-side target and an equal-parameter control; otherwise fold this row.**
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN pose/rate compositor. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/pose_fusion/`. Fire trigger: ps135 terminal and PZ4R scorer receipts both exist; rerank complete direct-v6/Stage-C-compensated candidates jointly with the HY1 head, never by isolated-delta addition.**
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN only. Consumer store: MAIN exact-evaluation receipt store. Fire trigger: a deterministic byte-closed hybrid strictly improves full-n600 local advisory S and passes custody/compliance; dispatch one exact contest replay on the exact archive bytes.**

## LIVE-HYPOTHESES

- The direct dense head can realize at least 82.824% of C1's Seg gain because the grammar expresses
  every changed cell and the frozen F26 coder prices the entire solved field at only +11 B; the
  remaining unknown is renderer/R/SegNet survival.
- Joint renderer adaptation can recover survival missed by direct token replacement because the
  terminal vehicle still exposes global semantic, basis, FiLM, and coefficient degrees of freedom,
  while ps135 proves the current receiver can be optimized beyond singleton starts.
- Event/wire enrichment makes a conditional dense-token/level-set fallback plausible if direct
  failures localize: C1 event sites are 225× enriched in ideal wire cost, and the top 10% of
  pair-patches contain 62.8% of events.
- PZ4R may reduce enough pose bytes to lower the Seg realization fraction needed for sub-0.15, but
  only a scorer-complete whole-object composition can test this.

## DEAD-ENDS

- Raw dense solved tokens plus generic Brotli q11 are closed at INSTANCE scope: 429,383 B is
  314,677 B above the shipped F26 token wire.
- Calling the C1 token plane a scored candidate is closed: grammar and causal decode pass, but
  learned-renderer, R, SegNet, and PoseNet survival are unmeasured.
- Adding +11 B directly to cp135 is closed: cp135 ships HP3 and a different probability object and
  token stream.
- Building a sparse hybrid before testing direct dense survival is closed: the entire dense solved
  plane already costs only +11 B on the calibrated F26 instance.
- Re-running entropy-only edge or pose contexts is closed at SR1's formulation scope: HPAC already
  captures them, with only -2 B for edge context and +43 B for pose context.
- Treating #580 as a scorer-null projector is closed: it is resize `range(A)` geometry and loses
  authority through integer/uint8 realization unless reverified.
- Reusing j11's one-dimensional Q3 split is closed: its useful nullity was zero.
- Reusing SQ2 uncap100 as a solved-paint route is closed: the n32 instance had 0/32 convergence and
  severe pose erosion.
- Pricing pose from the 19,221-byte PGQ1 envelope is closed: that envelope was nonrenderable and was
  superseded by the 183,137-byte PZ4R receiver object.
- Running a scorer or Modal beside ps135b is closed by the sole-lane contract; ps135b remains live
  in pass 3.
