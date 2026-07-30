# POINTER HONESTY — official competitive bar ~0.172141; custody-local 0.1910828242 [contest-CPU] UNMOVED

Everything in this memo is `[macOS-CPU advisory]`, `score_claim=false`. No
scorer, PoseNet, Metal, paid dispatch, live burn, or exact evaluator ran in this
arm. The QA24 burn-1 Seg endpoint is **UNKNOWN**. In particular,
`d_seg=0.00431179` is the old v4d anchor and MUST NOT be substituted for the
post-burn endpoint. No exact archive row moved either the official competitive
frontier or the local-custody anchor.

# ddm_su2 pose endgame program for the #782 post-burn slot

## 1. Verdict

The sole scorer slot should operate on one frozen, hash-bound post-burn archive
and in this order:

1. choose/rebuild the rate parent first (**WR1 Gate B is the primary
   target-capable alternative; Gate A is only a local-line fallback unless the
   same rebuild removes at least another 15,809 bytes**);
2. re-solve the geometric pose family on that exact parent, racing single-plane
   against the existing realizable two-plane form and only receiver-admitted
   experts;
3. run TT1 pose-only and joint modes, selecting realized best-of per pair;
4. refit/reselect QA66 photometrics with OFF available;
5. perform the terminal joint TT1 finish;
6. race representations on the final solved field, then exact recode,
   public-receiver parseback, n600 scorer replay, and full evaluator.

Gate A and Gate B are alternatives, not sequential deltas. Geometry, TT1,
pose6, and QA66 share actuators and must be refit/best-of selected, never added
as historical score deltas. SC1 is a coding/initialization fact, not a solved
actuator. P5-v2 is a cheap initializer whose quoted 194 bytes are only `s_t`.
PC1, QA68, and the distinct QA43 free-frame0 counterfactual do not enter the
slot without a concrete receiver-closed implementation.

This arm makes one bounded implementation ready: a real, shipped **v4d
warp-tail** solver/receiver in the existing PFS1/TT1/QA43-two-plane pool. It
does **not** implement or evidence the distinct free-frame0 counterfactual.

## 2. Stores consulted and receipt adjudication

Recall preceded design. The load-bearing stores were:

- `.omx/research/ddm_deferral_queue_ledger_20260729.md` (QA06, QA42, QA43,
  QA45, QA73);
- `.omx/research/ddm_wr1_reverse_waterfill_20260729.md`;
- `.omx/research/ddm_tt1_payload_gradient_tto_20260731.md`;
- `.omx/research/ddm_sc1_seeded_scene_carrier_20260728.md`;
- `.omx/research/ddm_p3v2_optimal_form_pose_resolve_20260729.md`;
- `.omx/research/ddm_pfs1_posefield_and_recompose_20260729.md`;
- `.omx/research/ddm_qa43_two_plane_parallax_20260729.md`;
- `.omx/research/ddm_ph3_realization_hybrid_adaptive_convocation_20260731.md`;
- `.omx/research/ddm_ja1_joint_atlas_waterfill_20260731.md`;
- `.omx/research/ddm_v4d_adaptive_hybrid_20260731.md`;
- `.omx/research/codex_findings_ddm_pc1_pose_stream_admission_20260724_codex.md`;
- `.omx/research/ddm_eg1_endgame_chain_20260728.md`;
- `.omx/research/ddm_bc1_qa24_compose_and_fire_20260731.md`;
- `.omx/research/codex_findings_ddm_fr1_fisher_actuator_base_curves_20260724T055154Z_codex.md`;
- `.omx/research/ddm_pp1_band_lemma_receipt_20260728.json`;
- `experiments/ddm_tt1_twin.py`, the v4d/PFS1 receivers, and the R7 coder.

| Asset | Authority and measured shape | Reach / wall clock | Pool adjudication |
|---|---|---|---|
| WR1 Gate A | MEASURED bytes: 274,333 B, rate 0.1826670842. Its old exact gate used stale pose parameters and regressed (`d_seg=.00553676`, `d_pose=.28002128`, `S=2.4097`); later re-solve overturned the pose part. | Rebuild/re-solve time not transferable. Rate alone exceeds 0.172141. | Alternative to B, never additive. Local-line fallback only unless further bytes are removed. |
| WR1 Gate B | MEASURED bytes: 174,578 B, rate 0.1162443243; never fired. | No realized d_seg/d_pose or wall clock. | Primary target-capable parent. Rebuild before every downstream pose fit. |
| v4d anchor | MEASURED n600: 360,238 B, `d_pose=.00858145`, pose term `.292941`, `d_seg=.00431179`, `S=.9639878179`. | Exact gate receipt; foreign pre-burn base. | Reference only. Its Seg endpoint cannot stand in for burn-1. |
| TT1 analytic twin | MEASURED n50 realized pilot, PROJECTED n600: best-of pose/joint `Delta S600=-.0630`, 13.6 min, projected pose term `.259293` (`d_pose≈.0067233`) on frozen v4c; bytes only approximately frozen. | 13.6 min pilot; final n600 duration not measured. | Production optimizer/acceptor. Same continuous pool as pose6, `(a,b)`, and QA66; rerun on final parent. |
| SC1 `e_p` rank-1 | MEASURED n600 structure: rank-1 energy `.9986`, AR-int5 2,039 B. Uncorrected painted base `d_pose=1.962`; no corrected endpoint. | Solve/recode wall not reported. | Representation/init race after the solve. Do not add 2,039 B or infer a d_pose endpoint. |
| P5-v2 | MEASURED stale n600 row: `d_pose=.3931`, pose term `1.9827`; 194 B is only the `s_t` stream and assumes `t_p` already exists. | Wall not reported. | Cheap warp initializer/fallback, same geometry pool. Never call 194 B the whole pose price. |
| PFS1 D1 | MEASURED n600 exact protocol: 6,864 B complete pose member, `d_pose=.22144216`, pose term `1.488093`, archive 569,996 B. | Exact evaluation was about 19 min; solve duration not isolated. | Fresh-parent fallback/init only; all parameters stale after a rate/base mutation. |
| QA43 two-plane warp | MEASURED on PFS1: 95/112 wins; pose term `1.4881→1.2630→.9127` (`d_pose=.0833`), total gain `-.5754 S`, at most 7.3 KB. | Wall not reported. | Same geometry pool as P5/PFS1/TT1. Refresh hard-tail order and solve on final parent. |
| QA43 free-frame0 top-112 | COUNTERFACTUAL: pose term `.382` (`d_pose=.0145924`) if the top 112 reach `1e-3`; the transferred `112×120+75=13,515 B` estimate is invalid for this content class. | UNBUILT; no wall clock. | Excluded. Requires a distinct receiver/content grammar. Falsifier remains whole-action price over 600 B/admitted pair. |
| QA68 expert menu | DERIVED/UNBUILT: 88 pairs hold 90% of v4c pose mass; estimated 1–3 KB; no archive or d_pose row. | No wall clock. | Same geometry/content pool. Admit only after receiver, selector, whole-archive pricing, and fresh-parent realized wins exist. |
| PC1 | BUILT typed grammar: 40 B zero home, +734 B nested action; first independent-row upper bound 5,014 B. Zero-home/no-descent worsened `S` by +2.17 and +16.65 on its two parents. | No successful descent. | Grammar/admission surface only; formulation remains open. Exclude from the one slot. |
| EG1 E3 | MEASURED n1 rehearsal: local `d_pose 5.281689→1.350459`; 107 B terminal section, not an outer archive. Later bounded absolute formulations plateau around `d_pose 10–38`. | n1 only. | Reuse its receiver/realized-acceptance contract. The fixed six-cosine implementation instance is dead; TT1 is the production optimizer. |
| QA66 photometrics | MEASURED: per-pair beta `d_pose .010384→.009533`, `-.013485 S`, +140 B. With dim0 and `(a,b)` refit, v4d reached `.008581`. | Same fit pass; isolated wall not reported. | Refit after geometry and TT1, with OFF per pair. Never add the historical delta to TT1. |

`MEASURED` above means only the cited object/base/axis. `PROJECTED`,
`COUNTERFACTUAL`, and `DERIVED/UNBUILT` values receive no score credit.

## 3. Exact ordered program for MAIN

| Stage | Action on one exact parent | Admission / pre-registered falsifier | Durable output |
|---:|---|---|---|
| 0 | Wait for burn completion; freeze exact archive bytes, SHA-256, receiver sources, target/scorer hashes, hardware axis, and burn Seg endpoint `x`. | Any parent/source/hash drift invalidates every downstream fit and restarts at Stage 0. Never substitute v4d `x=.00431179`. | Immutable base receipt. |
| 1 | Scorer-free rebuild of WR1 alternatives on the frozen tokens. Primary = Gate B; Gate A only if B fails closure or A already includes at least 15,809 additional bytes of shrink. Choose **A XOR B** before pose work. | Reject malformed/public-wire drift. A at 274,333 B cannot cross the official bar at nonnegative distortion. If B fails exact joint action, fall back honestly; do not splice A and B deltas. | Selected-parent archive and exact byte/hash receipt. |
| 2 | Fresh n600 parent Pose replay and hard-tail order. Race single-plane vs realizable two-plane and only concrete admitted experts. Use PFS1/P5 as initialization, not quoted payoff. The built warp-tail ladder is nested `k=56→112→200`. | Stop/refit at first nested stage with: no admitted pair; whole archive action over 600 B/admitted pair; non-improving archive-level pose+rate action; inactive pair/frame1 mutation; SMEVR/Brotli semantic disagreement; or public receiver mismatch. Negative scope = exact parent × basis × integer lattice × selected top-k, not family. | Atomic baseline/order/per-pair checkpoints, distinct stage archives and receipts. |
| 3 | TT1 **pose-only** pass on the Stage-2 winner. | Accept a pair only on realized uint8 Pose6 improvement versus the exact parent; otherwise retain Stage 2. | Per-pair best-of ledger and checkpoint. |
| 4 | TT1 **joint** pass from the pose-only state. | Accept only where the exact pose+other-authoritative terms+bytes action beats pose-only. Never sum the old `-.0630`. | Joint/pose-only per-pair selector and checkpoint. |
| 5 | Refit QA66 `(a,b)`, beta, and any retained photo coordinate after geometry/TT1; OFF is an explicit expert. | No realized improvement or any authoritative-term spill selects OFF. Historical QA66 values are not transferable. | Photo selector/member receipt. |
| 6 | Terminal TT1 joint finish over the selected geometry/photo state. | Bounded 2–3 relinearizations; realized monotone acceptance only. A plateau kills only this formulation/parent. | End-of-stage checkpoint, not loop-end-only state. |
| 7 | Representation race on the same solved semantic field: existing field/R7 versus SC1-style rank-1/AR-int5 where applicable. | Compare complete archive bytes and require byte-identical public decode. No disconnected 2,039 B proxy wins. | Exact candidate variants and representation receipt. |
| 8 | Recode, direct public receiver parseback, full n600 Pose/Seg replay, then `upstream/evaluate.py` on the exact winning bytes. | Candidate must beat the selected parent and then the official ~0.172141 bar. CPU/CUDA stay separate. Anything else leaves the competitive frontier unmoved. | Exact archive/SHA/bytes, logs, hardware custody, component metrics, evaluator row. |

The scorer slot should not be spent on QA68, PC1, the fixed EG1 cosine
instance, or the unbuilt free-frame0 counterfactual. Their missing receiver or
realized endpoint cannot be repaired by a longer run.

## 4. Composed-score arithmetic

Let:

```text
x       = exact burn-1 d_seg (UNKNOWN)
kappa   = 25 / 37,545,489 = 6.658589531221714e-7 score/B
delta_i = signed whole-archive bytes relative to gate i's existing pose member
S_i     = 100*x + sqrt(10*d_pose) + rate_i + kappa*delta_i
rate_A  = 0.18266708418686464
rate_B  = 0.11624432431816244
```

The table fixes `delta_i=0`; it is a reference matrix, not a composition
claim. `C_i` means `S_i=100*x+C_i`.

| Pose outcome | Status/base | pose term | `C_A` | `C_B` | maximum `x` to beat `.9639878179`, A / B |
|---|---|---:|---:|---:|---:|
| ideal zero pose | bound only | 0 | .182667084187 | .116244324318 | .007813207337 / .008477434936 |
| TT1 best-of | PROJECTED, foreign v4c | .259293173438 | .441960257624 | .375537497756 | .005220275603 / .005884503201 |
| v4d pose | MEASURED, foreign pre-burn | .292941120364 | .475608204551 | .409185444682 | .004883796133 / .005548023732 |
| QA66 floor | MEASURED, foreign base | .308762892815 | .491429977002 | .425007217133 | .004725578409 / .005389806008 |
| QA43 top-112 | COUNTERFACTUAL | .382000000000 | .564667084187 | .498244324318 | .003993207337 / .004657434936 |
| QA43 two-plane | MEASURED PFS1 | .912688336728 | 1.095355420915 | 1.028932661047 | impossible |
| PFS1 D1 | MEASURED PFS1 | 1.488093276646 | 1.670760360833 | 1.604337600964 | impossible |
| P5-v2 | MEASURED stale row | 1.982674960754 | 2.165342044941 | 2.098919285072 | impossible |

Necessary target conditions:

- Gate A rate alone exceeds 0.172141 and 0.15. Even at zero distortion it
  needs at least **15,809 B** removed to reach the official bar, or **49,061 B**
  to reach 0.15.
- Gate B versus 0.172141 requires
  `100*x + sqrt(10*d_pose) < .05589667568183755`; therefore
  `x < .0005589667568183755`, and at `x=0`,
  `d_pose < .0003124438352280530`.
- Gate B versus 0.15 requires
  `100*x + sqrt(10*d_pose) < .03375567568183756`; therefore
  `x < .0003375567568183756`, and at `x=0`,
  `d_pose < .00011394456407373995`.

Every non-ideal numeric pose term in the table already exceeds 0.172141.
Consequently none is a target-crossing endpoint even at zero Seg and zero
rate. These receipts rank mechanisms; a new joint descent must reach a much
lower pose term on the final burn/rate parent.

Byte counterfactuals demonstrate why complete archive pricing matters:

- QA66 `+140 B` changes `C_A/C_B` to
  `.491523197255/.425100437386`.
- QA43 `112×120+75=13,515 B` changes `C_A/C_B` to
  `.573666167938/.507243408070`, but this transfer is explicitly invalid for
  the required free-frame0 content class.

## 5. BUILT + TESTED in this arm

New code:

- `experiments/ddm_su2_qa43_tail_solver.py`;
- `experiments/inflate_runner_v4d_qa43_tail.py`;
- `experiments/test_ddm_su2_qa43_tail_solver.py`.

The generic harness is strict n600 and contains:

- fresh exact-parent Pose replay and deterministic hard-tail ranking;
- nested, atomic, resumable `k=56,112,200` checkpoints;
- bounded 2–3-relinearization integer GN with no RNG;
- complete SMEVR/Brotli candidate archives and exact 75-byte pair map;
- canonical R7 parse/re-encode, outer-ZIP parseback, source/adapter/target/scorer
  hash binding, and complete byte accounting;
- same-codec empty-tail controls separating fixed repackage/grammar overhead
  from marginal tail bytes;
- full-n600 null/inactive/active/frame1/codec semantic closure at every stage;
- the stated >600 B/admitted-pair falsifier priced by **whole archive action**,
  not candidate-minus-null marginal bytes;
- SSD storage preflight, atomic fsync+rename state, exact-prefix resume, and
  preserved stage archives.

The concrete adapter/receiver is honest but deliberately scoped:

- it consumes a signed-int4 `[active_pair,1,6,1]` R7 tail in the shipped
  receiver before the real v4d two-plane/photometric/rolling-shutter/uint8
  realization;
- it is the same PFS1/TT1/QA43-two-plane warp pool;
- it refuses `terminal-frame0`;
- validation uses deterministic **nonzero** coefficients, proves frame0
  changes, and compares the optimized adapter path with the exact public
  `Decoder(extracted_archive)` constructor.

Scorer-free verification:

```text
py_compile: PASS
ruff check: PASS
pytest: 9 passed in 35.40s
```

The real custodied validation used parent archive
`f1f3288062468e97c090ffe88ac81a6d6f76925743bd83aecb15307c0314a220`
(360,238 B) and returned:

| codec | candidate bytes | candidate SHA-256 | tail member | closure |
|---|---:|---|---:|---|
| SMEVR | 360,719 | `0fa354833b5b90a4ee7102f6a4457f227e1830f848c51f2739d22c28a031c8c6` | 196 B | nonzero frame0; canonical R7; all bytes consumed; ZIP + public receiver PASS |
| Brotli11 | 360,720 | `7c54940fe17c59fb6c8dac95a668db9b4aa9389a42ed6cba86a38a2bdc065754` | 198 B | nonzero frame0; canonical R7; all bytes consumed; ZIP + public receiver PASS |

This is a two-pair scorer-free receiver positive control, **not a pose result or
score**.

## 6. DESIGNED / BLOCKED, with no implementation credit

- The actual QA43 free-frame0 top-112 counterfactual remains
  `NO_VERDICT_IMPLEMENTATION_CUSTODY`: no distinct receiver/content grammar was
  built, so no command is offered and the family remains open.
- QA68 remains `DERIVED_UNBUILT`: no expert receiver, selector, whole archive,
  d_pose, or wall clock.
- PC1 remains a typed admission surface without descent.
- SC1 remains a representation hypothesis until the final solved field is
  encoded and compared as a complete archive.

These gaps are intentionally not papered over by relabeling the existing warp
solver.

## 7. Ready command for MAIN

First freeze/hash the final parent and verify it retains the v4d public receiver
grammar. Then replace the two explicit placeholder components below. This
command **will load the frozen PoseNet and consume the scorer slot**; it was not
run in this arm.

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/experiments" \
/Volumes/VertigoDataTier/pact/uv-envs/pact-main/bin/python \
experiments/ddm_su2_qa43_tail_solver.py solve \
  --program-kind warp-tail \
  --receiver-adapter __main__:create_v4d_warp_adapter \
  --adapter-arg parent_archive=/absolute/path/to/final_postburn_selected_rate_archive.zip \
  --adapter-arg receiver_deps_dir=/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1 \
  --top-k 56,112,200 \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_su2_qa43_FINAL_BASE_SHA256 \
  --relinearizations 3 \
  --damping 0.001 \
  --coefficient-limit 7 \
  --min-free-bytes 1073741824
```

Use `validate` with the same adapter arguments before `solve`. Do not change
`--program-kind` to `terminal-frame0`: the built-in adapter correctly refuses
that false claim.

## 8. Adversarial pass

| Risk | Attack and mandatory check |
|---|---|
| Base dependence | FR1 measured no deltas and concluded `NO_VERDICT_DATA_CUSTODY`; it supports neither dependence nor independence. Separately, WR1 stale-pose and PFS1 stale-frame incidents prove practical invalidation. Hash the selected parent and re-solve every pose/photo coordinate on it. |
| Band-lemma overreach | PP1 is position-coding economics for correction-support densities. It does not price pose fields, QA43 parameter tails, SC1, or WR1 drops. Do not use it to admit/reject these streams. |
| Counterfactual price transfer | The free-frame0 content class is not a six-parameter warp. Require a distinct receiver, realized uint8 effect, public parseback, and whole-action B/admitted pair. |
| Same-pool double counting | Geometry, TT1, pose6, and QA66 modify overlapping physical controls. Run sequential best-of/refit; never sum historical deltas. Gate A/B are XOR. |
| Stale hard-tail IDs | Recompute all 600 parent errors after the final rate/burn base. Any old top-112 list is only an initialization hint. |
| Archive overhead | Price parent→candidate whole action for the 600 B falsifier. Candidate→same-codec-null remains diagnostic marginal accounting only. |
| Adapter/public-wire divergence | Nonzero positive control must match `Decoder(extracted_archive)`, both codecs, exact frame1, exact pair map, and canonical R7. Stage closure checks all n600 semantics. |
| Wrong competitive target | The local 0.1910828242 row is custody context, not the competitive bar. Compare final exact bytes to ~0.172141. |
| Means narrated as end | A completed solver run that does not move the exact frontier is not goal progress. Report it as advisory and immediately route to the next exact-row path. |

## 9. LIVE-HYPOTHESES / DEAD-ENDS / NEXT-IF-RESUMED

### LIVE-HYPOTHESES

- Gate B plus a genuinely low post-burn Seg endpoint is the only banked WR1
  alternative with enough raw rate headroom to approach the official bar.
- Fresh-parent geometric best-of followed by TT1 and QA66 refit may beat every
  historical standalone pose row because it removes the measured staleness and
  composition confounds.
- The final solved field may retain SC1-like rank-1 compressibility, but only a
  complete same-output archive race can establish that.
- The distinct free-frame0 content family remains open and could dominate warp
  tails if a compact receiver grammar exists.

### DEAD-ENDS

- Unchanged Gate A as a competitive endpoint: rate alone is too high.
- Adding Gate A and Gate B deltas: they are alternative token states.
- Treating P5's 194 B as a whole pose member: `t_p` is assumed.
- Treating SC1 2,039 B as a solved d_pose row: no corrected endpoint exists.
- Shipping QA68 from a 1–3 KB estimate: no receiver or realized row exists.
- Reusing PC1 zero-home or fixed EG1 six-cosine as production finishers: the
  measured instances fail; their broader families remain scoped open.
- Calling the built warp-tail adapter the QA43 free-frame0 implementation.

### NEXT-IF-RESUMED

1. Read the final burn completion receipt and exact archive SHA; do not infer
   it from a PID or old v4d row.
2. Rebuild/select WR1 B (or a documented XOR fallback) before any scorer fit.
3. Run scorer-free `validate`, then the resumable command above.
4. At each nested stage inspect the holistic receipt: n600 d_pose, whole-action
   bytes, public-wire closure, inactive/frame1 identity, and falsifiers.
5. Continue TT1→QA66→terminal finish only from the selected stage checkpoint.
6. End with the exact public evaluator row or an explicit instance-scoped
   falsifier; MAIN review is mandatory before landing or dispatch.

Triality: the code is a pure-build receiver/measurement apparatus
(`[no-triality]`); this memo and the single canonical DAG FEED are the graph
leg; no new empirical equation is claimed beyond the existing contest-score
action and registered receipt laws.
