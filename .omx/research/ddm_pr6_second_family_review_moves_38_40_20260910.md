# ddm_pr6 — second-family review of pointer moves 38–40 and gs3 Addenda 19–20

`[no-triality] [p0-ledger-ok]` · owner `ddm_pr6` · 2026-09-10.
`research_only=true`; `score_claim=false`; review axis only.

## Verdict

The three `[contest-CUDA T4 n600]` scores reproduce from receipt components
within `2.78e-17`; the arithmetic P0 stop did not fire. The chain is exactly
move 37 → 38 → 39 → 40, and the canonical pointer is bound to move 40:
`S 0.13763861019288715 @ 180,233 B`.

The residual prediction is only partly right. All three residuals are entirely
the authority receipt's eight-decimal component printing, but only move 39 is
pose-print alone. Moves 38 and 40 contain both seg-print and pose-print terms;
calling either one “pose-print class” hides 23.9% and 26.8% of its residual.

Addenda 19 and 20 retain thirteen source-level misreadings. The central one is
decisive: “the token-level rate corner [is] near its entropy” is not supported.
BND2/BND3 report achieved code sizes—upper bounds—for named formulations, and
TC2 reports one fixed five-weight context extension plus a descriptive entropy
table. None measures an entropy lower bound or proves optimality over every
decoder-visible context or coder on the field. The source-supported conclusion
is: **every tested BND2/BND3 recode loses on move 37, and TC2's tested causal
extension saves 78 physical tail bytes; the coder/context family remains open.**

The three composition statements survive with a scope correction. The six lost
repairs are exactly located on four of the 13 shared pairs; the 78-token tail
cost rises from 4.8205128 to 5.1282051 bits/token, a 6.38298% anti-synergy; and
the composition's container search returns 3 B. That single return is one
favourable draw. “Two-sided” is established by FE1's earlier 133-draw
distribution, not by compose-39 alone.

PR #140 still carries move 23, so it is `40 - 23 = 17` pointer moves behind.
The current visible public page gained no new AI-attribution text since pr5;
the pre-existing body sentence naming Claude and Codex remains public.

## Exact score re-derivations

The primary receipts report passing T4 authority evaluations, empty validation
errors, and the same components mirrored at
`experiments/results/modal_auth_eval_mirror/contest_auth_eval_ddm_sj1_t4_token_predistortion_pass5_20260909.json:3-21`,
`contest_auth_eval_ddm_rp1_t4_rate_directed_predistortion_20260910.json:3-21`,
and `contest_auth_eval_ddm_sj1_t4_compose39_rp1_union_20260910.json:3-21`.
Their receipt SHA-256s are respectively `fbc62d6f…18339`, `41d24304…7b5f`, and
`cd6d5ef5…3d00`. Complete machine-readable arithmetic is retained at
`.omx/research/ddm_pr6_20260910/score_recomputation.json:1`.

Packet key: `move-38 packet` is
`.omx/research/ddm_sj1_t4_token_predistortion_pass5_20260909_pointer_move_38_20260910.md`;
`move-39 packet` is
`.omx/research/ddm_rp1_t4_rate_directed_predistortion_20260910_pointer_move_39_20260910.md`;
and `move-40 packet` is
`.omx/research/ddm_sj1_t4_compose39_rp1_union_20260910_pointer_move_40_20260910.md`.

`S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489`.

| move | receipt components `(d_seg, d_pose, B)` | derived terms `(seg, pose, rate)` | recomputed S | recomputed − receipt | result |
|---:|---|---|---:|---:|---|
| 38 sj1 pass 5 | `0.00010632, 5.06e-6, 180436` | `0.010632, 0.007113367697511495, 0.1201449260655521` | `0.1378902937630636` | `0` | PASS |
| 39 rp1 | `0.00010698, 4.89e-6, 180186` | `0.010698, 0.0069928534948188355, 0.11997846132727157` | `0.13766931482209040` | `+2.7755575615628914e-17` | PASS |
| 40 composition | `0.00010636, 4.89e-6, 180233` | `0.010636, 0.0069928534948188355, 0.1200097566980683` | `0.13763861019288715` | `0` | PASS |

The packets independently print the same term decompositions and custody at
move-38 packet:7-23, move-39 packet:7-23, and move-40 packet:7-23. The delta
chain also reproduces from the receipt scores:

| transition | exact ΔS | exact ΔB | packet match |
|---|---:|---:|---|
| 37 → 38 | `-2.7006274514573825e-5` | `+48` | yes, move-38 packet:18-23 |
| 38 → 39 | `-2.2097894097322657e-4` | `-250` | yes, move-39 packet:18-23 |
| 39 → 40 | `-3.070462920323758e-5` | `+47` | yes, move-40 packet:18-23 |

The cumulative move-37 → move-40 delta is
`-0.00027868984469103797`. The current pointer names move 40's lane, archive
SHA and score at `.omx/state/canonical_frontier_pointer.json:3-15`, repeats its
180,233-B CUDA anchor at lines 38-59, and reports no maturity refusals at lines
62-71.

## Projection residuals

Residual means `realized S - projected S`; every residual here is positive, so
every projection was optimistic.

| move | projection | exact S | residual | source decomposition | verdict |
|---:|---:|---:|---:|---|---|
| 38 | `0.13788255886142803` | `0.1378902937630636` | `+7.734901635580993e-6` | seg print `+1.848422e-6`; pose print `+5.886479e-6` (`ddm_sj1_multipass_token_predistortion_20260905.md:1867-1885`) | Entirely 8-dp print, but **mixed seg+pose**: 23.90% seg / 76.10% pose. Packet line 27's pose-class shorthand is incomplete. |
| 39 | `0.1376665464876166` | `0.13766931482209038` | `+2.768334473796097e-6` | projected seg is already the printed `0.00010698`; resolved pose is `4.886129e-6` versus printed `4.89e-6` (`ddm_rp1_rate_directed_token_predistortion_20260909.md:460-469,567-569`) | Entirely 8-dp **pose print**. Prediction confirmed. |
| 40 | `0.13763577081553857` | `0.13763861019288715` | `+2.839377348573535e-6` | seg print `+7.597907e-7`; pose print `+2.079587e-6` (`ddm_sj1_multipass_token_predistortion_20260905.md:2065-2081`) | Entirely 8-dp print, but **mixed seg+pose**: 26.76% seg / 73.24% pose. Packet line 27's pose-class shorthand is incomplete. |

The useful law is therefore “component-print residual,” not “pose-print
residual.” Any future projection should retain unrounded seg and pose components
and pre-register a bound for both printed fields.

## Addenda 19–20 source audit

The audited text is gs3 Addendum 19 at
`.omx/research/ddm_gs3_gestalt_after_submission_20260903.md:1085-1093` and
Addendum 20 at lines 1095-1105.

Citation key below: `gs3` is that gestalt memo; `bnd2`, `bnd3`, `tc2`, and
`rp1` are respectively `.omx/research/ddm_bnd2_boundary_segment_code_real_encode_20260910.md`,
`.omx/research/ddm_bnd3_address_term_decomposition_20260910.md`,
`.omx/research/ddm_tc2_lane_boundary_context_map_20260910.md`, and
`.omx/research/ddm_rp1_rate_directed_token_predistortion_20260909.md`.
`MISPREDICTED_CENSUS.json` is
`/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/retained/MISPREDICTED_CENSUS.json`.

| class | Addendum claim | source verdict and corrected statement |
|---|---|---|
| DIFFERENT-OBJECT / assignment-as-topology | A19: median assigned segment length 1 means the mispredicted population is intrinsically “isolated single cells,” falsifying a curve population (gs3:1089-1091). | BND2's greedy edge assignment creates charged starts at gaps, branches, class changes and offset changes; it explicitly says these are **assigned-segment lengths**, and changing assignment can change them (`ddm_bnd2_boundary_segment_code_real_encode_20260910.md:88-104`). Correct: median 1 is measured for two declared greedy assignments, not the population's intrinsic or globally optimized topology. |
| BOUND-DIRECTION | A19: “every address costs more than the token it replaces” (gs3:1091). | The best tested side channel costs 2,013 B and displaces 597 B on its 2,125-cell support (`ddm_bnd2...md:23-39`; `ddm_bnd3_address_term_decomposition_20260910.md:49-59`). An achieved losing code is an upper bound on achievable description length, not a lower bound on every address code. |
| DIFFERENT-OBJECT | A19: the masked control proves the 83-KB attribution is not detachable “in the strongest sense” because later causal contexts make the stream larger (gs3:1091). | The masked intervention changes 235,044 field symbols and the later probability trajectory, producing 250,892 new coder misses (`ddm_bnd2...md:149-161`). BND3's different counterfactual keeps the original field/trajectory and supplies every miss location **and true symbol** free, reducing 119,784 B to 36,525 B (`ddm_bnd3...md:61-81`). Correct: the 83,259-B oracle displacement is not a legal payload, while the masked field intervention is not evidence that unchanged-field surprise is physically inseparable. |
| BOUND-DIRECTION / formulation-to-family | A19: “the rate side is now measured closed for token-level boundary description” (gs3:1091). | BND2 closes six greedy formulations only and says its sizes are upper bounds, never entropy floors (`ddm_bnd2...md:163-176`). BND3 adds 48 losing formulations but expressly retains FORMULATION scope and corrects A19's broader wording (`ddm_bnd3...md:120-138`). |
| DIFFERENT-OBJECT / family closure | A19: every representation-level door that keeps the shipped token field is closed or vacuous; only generator and field remain (gs3:1093). | The sources exclude richer assignment, joint refits, alternative geometry/context maps and other probability models. Correct: the tested re-description formulations lose; no exhaustive representation-family census or lower bound exists. |
| ATTRIBUTION-AS-PHYSICAL-ALLOCATION | A20: BND2/BND3 show “the address term is only 1,717 B” (gs3:1103). | The 121,200-B packet has one shared Brotli block with no unique address/content allocation. `1,717 B` is a **separate transform** of the logical address stream for the best 2,125-cell support, not a physical component of the shared packet or a full-population address price (`ddm_bnd3...md:35-59`). |
| DIFFERENT-OBJECT / wrong mechanism | A20: the `+1,416 B` segment loss “is context disruption” (gs3:1103). | The lossless segment receiver inserts supplied symbols at their original causal positions and updates every corrector/mixer state (`ddm_bnd2...md:132-143`). Its loss is exactly `2,013 - 597 = 1,416 B` on that support (`ddm_bnd3...md:56-59`). Context disruption belongs to the separate masked-field intervention. |
| DIFFERENT-OBJECT | A20: “a zero-cost ORACLE flag would displace 83,259 B” (gs3:1103). | The oracle supplies **both** miss locations and true symbols. A miss/not-miss flag does not identify which non-modal class to emit (`ddm_bnd3...md:63-81`). Call it a free support-and-symbol oracle, not a flag. |
| ATTRIBUTION-AS-PAYLOAD / BOUND-DIRECTION | A20: TC2's GT-edge context “buys 5,480 B,” supporting a geometry ceiling (gs3:1103). | `5,480.397216 B` is a sum of realized ideal codelength reductions for a fixed five-weight extension, not serialized payload bytes (`ddm_tc2_lane_boundary_context_map_20260910.md:18-34`). The only physical result is a 78-B counted-rider saving, not a ZIP delta (`tc2:97-127`). The GT-edge feature is noncausal and explicitly not a universal upper bound (`tc2:69-95`). |
| SCREENED-SUBPOPULATION-AS-POPULATION | A20: “87.6% of the payable bits” are on stored-token/GT agreements (gs3:1103). | The census keeps 643,198.406 of 666,069.059 mispredicted bits—96.566%—using a `>=0.25` best-alternative-saving screen (`MISPREDICTED_CENSUS.json:300-311`). `563,726.621 / 643,198.406 = 87.644%` is the **screened** share (`MISPREDICTED_CENSUS.json:2-19`). With the 22,870.653 omitted bits unresolved, the full-population share is only bounded to **[84.635%, 88.069%]**. The separate 4.36% neutrality result is full-n600 over 19,200 realized proposals (`rp1:377-388`) and does not repair this census denominator. |
| PARTIAL-SCREEN-AS-POPULATION | A20: “neutrality is 4.36% and flat in rank” is carried as one population law (gs3:1103). | `4.36%` is the all-600-pair top-32 pass (`rp1:377-388`), but the five-stratum flatness claim, including ranks 32–400, comes from only 384 proposals on 12 seeded pairs (`rp1:95-115,136-154`). The source gives the rank-100–400 estimate ±38% uncertainty. Correct: full-n600 top-32 neutrality is 4.36%; cross-rank flatness is a seeded n12 signal, not an n600 law. |
| FORMULATION-TO-FAMILY / ENTROPY OVERCLAIM | A20: every field-built context has made the boundary surprise “near-inherent,” it is “near its entropy,” and no coder over the field can reach the corner (gs3:1095,1103). | BND2/BND3 tested 54 recodes and explicitly refuse a family floor; TC2 tests a fixed feature bank and says alternative maps are non-nested and its entropy buckets cannot be a general ceiling (`tc2:42-67,69-95`). **No source measures an entropy lower bound.** The missing measurement is a full-n600 lower bound or optimality certificate for a precisely declared decoder-visible context/model class, with every model/side-information byte charged, paired with a real serialized coder gap. Without a declared model class, “any coder” is not empirically falsifiable. |
| FORMULATION-TO-EXHAUSTIVE-CLASS LIST | A20: exactly three classes remain and the generator is the only representation-level door (gs3:1105). | BND2/BND3 explicitly leave richer joint assignments and probability models open; TC2 excludes joint refitting and alternative geometry, and its already-fired successor tests a better causal predictor (`.omx/state/canonical_task_status.jsonl:844-852`). Correct: field edits, a better causal predictor, and a generator are named live classes—not an exhaustive proof that every other representation class is closed. |

No contiguous-prefix-as-population error remains in Addenda 19–20. Addendum 20
correctly distinguishes the 84 edited-pair frame-0 screen from the all-600
untouched-field measurement; the source reports 74/84 versus 78/600 and default
mode best on 502/600
(`ddm_sj1_multipass_token_predistortion_20260905.md:1899-1919`). Its separate
rank-flatness law nevertheless promotes a seeded n12 screen, as corrected in the
table. The screened-census row is the fifth requested misreading class.

One separate arithmetic error is outside the five classes: A20 says moves
31–39 improved `-2.5e-3`; the exact endpoints are
`0.1398140172839628 - 0.13766931482209038 = 0.0021447024618724275`
(`ddm_sj1_t4_token_predistortion_joint_20260906_pointer_move_31_20260906.md:1-20`;
move-39 packet:13-23).

## Composition-law check

### Seg is sub-additive on this overlap — verified

The composition receipt lists 13 shared pairs—`7,118,165,167,202,237,293,398,
502,546,565,584,593`—and zero token-position collisions
(`/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price/compose39/COMPOSE.json:3-21`).
The original move-38 seg ledger gives repairs `1,2,1,1` on pairs
`237,293,398,565` (`SEG_CREDIT.json:1427-1431,1763-1767,2393-2397,3395-3399`).
On the move-39 base they become `0,0,0,-1`; the final receipt names the three
zero pairs, the one harmed pair, and 74 total repairs
(`/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEGCHECK_composed.json:7-15,108-176,276-281`).
Thus `5 - (-1) = 6` repairs are lost, and every affected pair is in the 13-pair
overlap. This verifies the law for this composition instance, not every token
composition.

### Rate anti-synergy — verified for the 78-token splice

On move 37 the subset costs 47 stream/archive bytes, or
`4.82051282051282` bits per changed token
(`/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price/PRICE_subset.json:15-25,39-47`).
Rebased on move 39, its pre-close cost is 50 B, or
`5.128205128205129` bits/token
(`/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/PRICE_candidate.json:15-25,39-47`).
The ratio increase is `(5.128205128205129 / 4.82051282051282 - 1) =
6.3829787%`, matching the packet's 6.4% shorthand. It is an instance-level
adaptive-coder interaction, not a constant tax.

### Container two-sided — compose receipt proves the favourable side only

The staged composition is 180,236 B and the closed archive is 180,233 B, so
the container search returns 3 B
(`/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/STAGE_TAIL.json:3-14,25-35`;
`candidate/CLOSE.json:3-11,36-39`). Relative to move 38's exact +48-B archive
delta, the final composition's +47 B beats the naive byte sum by 1 B. The
compose receipt alone does not prove both signs. FE1 supplies that broader
evidence: among 133 real single-move archive draws, 67 shrink, the range is
`-61..+98 B`, and mean/sd are `+0.1/34.8 B`; no 117-move continuation beats the
best single (`ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md:537-575`).
Therefore “container lottery is two-sided” is supported by the inherited FE1
law, while compose-39 is one favourable sample.

## PR #140 status

The public PR carries AFR1's 180,002-B move-23 archive
(`ddm_fs2_pointer_move_25_20260904.md:43-46`), while the current packet is move
40. Its lag is therefore **17 moves**. Publication remains operator-gated.

The current public-page audit is retained at
`.omx/research/ddm_pr6_20260910/pr140_public_audit.md:1-21`. It found no newly
added AI-attribution sentence since pr5, but the pre-existing public body still
says “I used coding agents (Claude as orchestrator of Codex subagents)”
(`experiments/results/ddm_fr2_final_review_20260903/pr_body_FINAL_POSTED.md:68-75`).
The narrower commit-history statement—no co-author or AI-attribution trailer—is
recorded at `.omx/research/ddm_pr140_submission_posted_20260903.md:24-28`. No
public write occurred.

## RECALL EVIDENCE

The charter and common contract were read in full. The required governing and
source floor was also read: PROGRAM, AGENTS/CLAUDE (byte-identical copies),
operating handoff, live hot state, canonical pointer, the three move packets and
authority receipts, sj1 §§26–33, rp1, BND1/BND2/BND3/TC2, gs3 Addenda 19–20,
and pr5's reference review and public audit.

Independent content searches covered `.omx/research/`, retained SSD/AP receipts,
the canonical equations export, research index/DAG, design/SPEC corpus, task
ledger and lane registry. Queries included `boundary segment|token entropy|
lane context|near entropy`, `83,259|address|oracle|masked`, `87.6|screened|
neutrality`, `compose39|shared pairs|lost repair|anti-synergy|container lottery`,
and `PR #140|move 23|AI attribution`. The canonical equations export was
searched for `boundary|tail|context|entropy|pose|composition|predistortion`.

Beyond the charter's seeds:

- FE1's 133-draw distribution changed the container verdict from “one
  favourable compose draw” to a supported, narrowly scoped two-sided law
  (`ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md:537-575`).
- The task ledger shows TC2's better-predictor and seal-intake follow-ons have
  already **FIRED** into TC3 (`.omx/state/canonical_task_status.jsonl:844-852`).
  TC3 then detected move 40 and began a current-field rebind; this review folds
  any duplicate predictor dispatch rather than orphaning one.
- BND3 itself says A19's broader all-representation wording must retain
  formulation scope (`ddm_bnd3_address_term_decomposition_20260910.md:135-138`),
  and TC2 explicitly rejects its GT map as a universal ceiling (`tc2:89-95`).
  Those source-owned exclusions changed the prior “near entropy” claim to an
  over-generalization, not a clean review.
- The index/DAG/design search found older near-entropy statements on different
  token fields and coder objects, but no move-37 lower bound or optimality
  certificate that closes the source gaps above. No number transferred.

## Boundaries and dispositions

This was a read-only review plus memo/evidence write. No new measurement,
pricing, scorer, archive build, Modal dispatch, pointer mutation, public
submission mutation, code edit, or `upstream/` write occurred. Existing exact
rows remain `[contest-CUDA T4 n600]`; every new conclusion is receipt-derived
and `score_claim=false`.

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer
  `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md` plus
  `.omx/state/canonical_task_status.jsonl`; fire when MAIN harvests this memo:
  append Addenda 19–20 corrections for mixed component-print residuals,
  formulation-only BND2/BND3 closure, the oracle/support distinction, TC2
  codelength-versus-payload, the screened 87.6% denominator, the absent entropy
  lower bound, and the exact moves31–39 delta.
- **FOLDED** — owner `ddm_tc3`; consumer existing TC3 retained store
  `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/`; fire trigger was
  the already-consumed TC2 `BETTER_PREDICTOR` and `SEAL_INTAKE` orders. Do not
  dispatch a duplicate; TC3 owns the move-40 rebind and final typed verdict.
- **FOLDED** — owner operator; consumer existing
  `ddm_ps2_pr140_update_operator_decision_gate_20260904` in
  `.omx/state/canonical_task_status.jsonl`; fire only on an explicit one-line
  publication decision. This review neither infers authority nor changes PR
  #140.

## LIVE-HYPOTHESES

- A better causal lane predictor may still yield a small current-field rate win
  because TC2's simple causal extension saved 78 physical tail bytes and left
  joint refitting and richer past-only geometry outside its bound; TC3 is the
  active owner testing that hypothesis on move 40.
- A jointly optimized boundary grammar may still beat the tested segment codes
  because BND2/BND3's 54 achieved encodes are upper bounds and explicitly omit
  global assignment, richer joint support, and other probability models.
- A generator that changes the object may avoid transmitting the incumbent
  field's expensive surprises because no field-preserving coder tested here
  approaches the rate corner; this remains plausible, but none of these
  receipts prices or constructs it.

## DEAD-ENDS

- Calling moves 38 or 40 “pose-print only”: their source decompositions include
  nonzero seg-print residuals.
- Treating greedy assigned-run length as intrinsic boundary topology: the
  assignment itself cuts paths at branches, class and offset changes.
- Treating BND2/BND3 achieved lengths as entropy floors or universal address
  lower bounds: achieved codes establish the opposite inequality.
- Calling the `1,717 B` separate address transform a physical allocation of the
  121,200-B shared packet or a whole-population address price.
- Calling the BND2 loss context disruption: `2,013 - 597 = 1,416 B`; the
  context-disrupting masked field is a different counterfactual.
- Calling BND3's free support-and-symbol oracle a zero-cost flag: a flag does
  not identify the non-modal symbol.
- Calling TC2's 5,480 ideal codelength reduction serialized bytes or a geometry
  ceiling: its physical result is 78 tail bytes and its map is noncausal.
- Reporting 87.6% as a full-population GT-agreement share: it is measured on a
  96.566%-of-bits screened census; the full share is only bounded.
- Reporting cross-rank neutrality as an n600 law: ranks 32–400 were supported
  only by the seeded n12 screen, including seven neutral events in the highest
  rank stratum.
- Reporting the move-37 field as near its entropy or unreachable by any coder:
  no entropy lower bound or all-coder optimality certificate was measured.
- Re-dispatching TC2's better-predictor work: its fire orders are already
  consumed by TC3.
- Reporting PR #140 at any lag other than 17, as newly AI-attributed since pr5,
  or as free of AI disclosure: it carries move 23, gained no new attribution
  since pr5, and retains the pre-existing Claude/Codex body sentence.

Own-vehicle frontier, **unchanged by this review**: **S 0.13763861019288715 @
180,233 B [contest-CUDA T4 n600]**, archive SHA-256
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
