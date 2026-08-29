# ddm_fcd3 pose-screened re-selection — retained heavy-tail screen and byte-real rung ladder

**Date:** 2026-08-29  
**Task:** #1320  
**Owning charter:** `.omx/research/charters/ddm_fcd3_pose_screened_reselection_20260829.md`  
**Status:** `TERMINAL_INSTANCE_REFUSED_SEG_BAND`  
**Axis:** `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; `promotable=false`

## Result

The retained fcd2 population has the predicted heavy pose tail, and pair screening preserves real
coder credit. The byte-real best rung is `tau_1e-6`: 450/600 pairs, 4,194/5,268 exact B positions,
and a 177,227 B body, `-2,965 B` versus jt21. A fresh solve on that exact body repaired pose and
published repeat-identical archive bytes at 177,252 B, still `-2,940 B` versus jt21, with
`d_pose=5.8495951113985735e-6 <= 6.3656845167356244e-6 + 1e-8` on the publish instrument.

The ordered full n600 frozen-scorer comparison then refused the instance. Against the same-run
jt21 base, realized `d_seg` moved from `0.0003474002587608993` to
`0.0003874630492646247` (`+4.006279050372541e-5`), while realized `d_pose` improved by only
`-8.065981091931462e-7`. The exact `-2,940 B` rate gain and pose gain do not pay for that Seg
regression: recomputed advisory S moved from `0.19306448449174635` to
`0.19500780888250857`, net `+0.0019433243907622244`. This is
`INSTANCE-REFUSED-SEG-BAND`; no seal, READY row, Modal dispatch, or pointer mutation is permitted.

## RECALL EVIDENCE

I searched the research corpus by content for `field-for-coder`, `pose screen`, `pose-safe`,
`re-selection`, `Schur`, `real re-encode`, `marginal price`, `GT lineage`, and `pose tail`; inspected
the canonical research indexes and `sub015_DAG_*` FEED blocks, design/spec surfaces, canonical task
status, the named harness-bridge surface, and the live lane ledger; and ran
`tools/list_canonical_equations.py --json`.

Beyond the charter's named seeds:

- `ddm_ps2_pose_projection_nscaling_20260818.md` says a selection that moves pose must be ranked on
  the joint objective, not the Seg axis alone. This changed the rung choice from rate-only to the
  known rate-plus-screened-pose components while leaving realized Seg explicitly unknown.
- `ddm_fs3_jg5_real_price_reopen_20260820.md` measured a +223 B marginal member at 5.9467 bits/token,
  2.24x its favorable set average. Together with `token_rate_model_direction_dependence_v1` and
  `greedy_set_average_vs_marginal_price_v1`, this required real full-stream re-encoding of all rungs.
- `ddm_tv2_evaluator_tolerance_curve_20260824.md` found Seg slack and pose damage co-located on its
  object. That supplied the formulation-level falsifier: a gate-clean rung with under 45 B credit
  would close pair-level re-selection on this body rather than invite another entropy projection.
- `ddm_iv1_inversion_pose_actuator_20260818.md` and
  `ddm_na10_negative_audit_fresh_laws_20260819.md` reinforced the base-error-dependent hard pose
  tail and the 13.4x population variance law. I therefore reported concentration over all 600 pairs,
  not a prefix or a mean alone.
- Relevant registered laws were `score_marginal_lagrange_multipliers_v1`,
  `pairset_component_marginal_score_decomposition_v1`,
  `token_rate_model_direction_dependence_v1`, `greedy_set_average_vs_marginal_price_v1`,
  `compensated_semantic_edit_exchange_v1`, and `section_coding_axis_closure_v1`.
- I did not find task #1320 in the exact searched scopes `.omx/state/canonical_task_status.jsonl`
  and `.omx/state/harness_tasklist_bridge_20260803.jsonl`; the latter path is absent in this checkout.
  This is bounded absence in those scopes, not a global nonexistence claim. The charter, current
  commit, main hot state, and fcd2 memo remained route authority.

The recall changed the ranking arithmetic and made the concentration curve load-bearing. It did not
change the preregistered thresholds, exact identity control, fresh-solve requirement, publish band,
or scorer ordering.

## Trigger, custody, and lane

- All payloads and receipts are retained under the existing consumer store
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/fcd3_pose_screened_reselection/`.
- The joined benefit pool is 27,105 B, sha
  `cc09fd9d4cb9a7253df30dbe38d5f60e33ee9e62c8217d9d0b1276ea5c2b5042`.
- The base archive is 180,192 B, sha
  `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3`.
- fcd3 claimed `ddm_fcd3_scorer_20260829` at `2026-08-29T18:55:03Z`. The newest r8/r9/r10 scorer
  rows were terminal before this claim; their older prelaunch rows were not treated as live jobs.
- After both sequential scorer receipts sealed, the lane was terminal-closed as
  `refused_seg_band`; no scorer job remains active under this lane id.
- The retained fcd1 n600 inverse-coder control was copied into the fcd3 subtree with its source
  receipt. It is an exact same-body reuse, not a fresh run: 113,601 B, sha
  `4c9dc10c0746e1f3bbaed1b754544fbc8ab4b981bbdb37136dc3076cdb976ba7`, byte-identical to jt21.
- Storage preflight observed about 30 GiB free before materialization. Every candidate field, edit
  plane, coder stream, archive, checkpoint, public decode, and receipt was retained; no measured
  payload was discarded.

## Retained per-pair screen

The join is exactly 600 pairs. The benefit pool has 5,268 positions on 555 active pairs. All five
GN banks and all five refinement banks cover 600 unique pair ids with no duplicate or missing pair;
the lower measured `final_d_pose` per pair is the screen value. The eight uncompensated measured-
improved pairs were force-kept at every rung: `[60, 102, 138, 360, 361, 380, 486, 554]`.

Best-bank provenance is 319 refinement rows and 281 GN rows. Selected-row stop reasons are 317
`no_improving_step`, 281 `gn_bank_best_pre_refinement`, and 2
`converged_below_materiality_floor`. The full refinement population itself contains 597
`no_improving_step`, 2 `converged_below_materiality_floor`, and 1 `lattice_floor`.

The complete 600-row table is retained as `screen/per_pair_screen.jsonl`; vectors and masks are in
`screen/screen_vectors.npz`.

### Concentration of positive pose excess

| Worst k pairs | uncompensated fraction | best-compensated fraction |
|---:|---:|---:|
| 1 | 3.1142% | 14.4139% |
| 5 | 13.9347% | 47.8884% |
| 10 | 23.5308% | 68.3568% |
| 25 | 40.6113% | 89.1377% |
| 50 | 56.8186% | 97.8964% |
| 100 | 75.3320% | 99.9859% |

After compensation, 6/16/27/38/60 pairs capture 50/80/90/95/99% of positive excess. Before
compensation the corresponding counts are 39/120/184/244/356. The preregistered heavy-tail
prediction is therefore supported on the full n600 population.

## Screen ladder and real joint re-encode

Ties at the exact per-pair threshold are dropped; the eight measured-improved pairs are then forced
in. Every row below is a fresh full-stream encode through the fcd1 path, not an entropy estimate.

| Row | threshold | pairs | B positions | archive | delta vs jt21 | rate delta S | retained-screen `d_pose` | known-axis projected delta S, Seg excluded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full-set identity | n/a | 555 | 5,268 | 176,436 B | -3,756 B | -0.0025009662279268756 | screen control | n/a |
| `tau_1e-8` | `1e-8` | 359 | 3,332 | 177,856 B | -2,336 B | -0.0015554465144933924 | `5.822830100262725e-6` | -0.0019032234310411842 |
| `tau_1e-7` | `1e-7` | 408 | 3,842 | 177,481 B | -2,711 B | -0.0018051436219142067 | `5.826306705543207e-6` | -0.002150642854206858 |
| `tau_1e-6` | `1e-6` | 450 | 4,194 | **177,227 B** | **-2,965 B** | **-0.001974271796007238** | `5.8507580495174615e-6` | **-0.002303770985988703** |

The full-set candidate archive is byte-identical to fcd1 union sha
`c45ab4e687d1a598b2c2191e5c4bf176bb1c12b24748795434cd109eb9a3aa6b`. This closes the subset-
filter identity control. The byte-real winner is `tau_1e-6`: the extra 629 B of rate credit versus
`tau_1e-8` is worth more than the retained-screen pose difference. Realized Seg is unknown here and
was not inferred from B labels.

Machine-readable receipts are `screen/SCREEN.json` and
`reencode/REAL_REENCODE_SUMMARY.json`. The selected body archive sha is
`a08ba488ba8dddf48e763420e46b43adf470baf4a2fb9df763a051639abce014`.

## Fresh exact-object Schur chain

The selected 177,227 B body was decoded through its public receiver before any solve. The retained
3,662,409,600 B raw has sha
`383c39f1e57f3f66dd15420da47c6806a805fe7fdfad33c55353132ded0df2af`. The fresh chain then used
five disjoint 120-pair GN shards and five disjoint 120-pair diminishing-returns shards.

| Stage | n600 result | Archive result | Disposition |
|---|---:|---:|---|
| candidate-bound baseline | uncompensated `d_pose=0.0010807806388255527`; same-instrument jt21 `6.3656845167356244e-6`; 442 worse / 8 better pairs | exact 177,227 B body | fresh-object binding passed |
| GN close | 448 improving/admitted; `d_pose=5.880355139506986e-6` | 177,250 B, sha `0cd28e9a5e7e8dd28d89e5b07d776b49c18178634732144fc197981c39a399e3` | publish band already met; refinement still required |
| diminishing-returns refinement | 598 `no_improving_step`, 1 `lattice_floor`, 1 `converged_below_materiality_floor`; no budget stop | 600 retained terminal rows | locally exhausted at configured optimal form |
| final close | 448 admitted; `d_pose=5.8495951113985735e-6` | 177,252 B, sha `a4913f44d261d5272fc2b83dffdcad1bf5e4b757c648e2d8207c3eb7f428f6ac` | candidate close |
| deterministic repeat | identical `d_pose` | byte-identical 177,252 B, same sha `a4913f44...` | repeat identity passed |
| in-code publish gate | `5.8495951113985735e-6 <= 6.375684516735624e-6` | pin-consistent runtime `runtimes/published_tau_1e-6/` | **PUBLISHED** |

The publish receipt is `fresh_solve/tau_1e-6/schur/publish_refined/PUBLISH.json`; close receipt
shas are `179059e4ee532fbabf2c51bec424f25acc3e97aeda1af15325df837be8a6345c` and
`710ef26fb9d1854690a5f7970def6df7ebe2e1d9f677dbfed22efbd83eddd79a`. The published archive was
decoded independently before scoring. Its retained raw is 3,662,409,600 B, sha
`432be791a854c900eedb404cd5292e053015b9f39670e3f8fdf153f2d7170988`.

Thus the retained-bank `5.8507580495174615e-6` was only a useful screen: publication rests on the
fresh exact-object value and repeat-identical compiled bytes, not on carried fcd2 compensation.

## Full scorer table and score arithmetic

The base and candidate ran sequentially through the same frozen scorer closure, seed 1234, four CPU
threads, upstream batch size 16, 38 atomic stages (37x16 + final unpadded 8), and exact realized
uint8 raws. Both receipts are `[macOS-CPU deterministic batch16 scorer advisory]`,
`score_claim=false`, `promotion_eligible=false`, and `pointer_mutation_allowed=false`.

| Object | Archive bytes | realized `d_seg` | realized `d_pose` | `100*d_seg` | `sqrt(10*d_pose)` | rate term | recomputed S |
|---|---:|---:|---:|---:|---:|---:|---:|
| jt21 base | 180,192 | `0.0003474002587608993` | `0.0001470109127694741` | `0.03474002587608993` | `0.03834200213466611` | `0.1199824564809903` | `0.19306448449174635` |
| published `tau_1e-6` | 177,252 | `0.0003874630492646247` | `0.00014620431466028094` | `0.03874630492646247` | `0.03823667279723498` | `0.11802483115881111` | `0.19500780888250857` |
| candidate minus base | **-2,940 B** | **`+4.006279050372541e-5`** | `-8.065981091931462e-7` | **`+0.004006279050372541`** | `-0.00010532933743113287` | `-0.0019576253221791837` | **`+0.0019433243907622244`** |

The measured delta is far outside the canonical `+/-3.5e-6` admission band on the wrong side.
The rate credit is real and pose is slightly better, but realized Seg spill dominates both. The
per-pair scorer stages show the Seg regression is diffuse: 404 pairs worsen, 31 improve, and 165
tie; the worst ten pairs contain only 6.75% of positive Seg damage. That supporting census does not
replace the receipt's float32 aggregation DAG; the table above uses the aggregate receipt values.

Receipts are `full_scorer/base_jt21/11_batch_replay_receipt.json` (receipt sha
`256d0dacbf19ecf96ac5d54a2386d5fa2c0c026c3d7e4f2828198dd4796b2435`) and
`full_scorer/tau_1e-6/11_batch_replay_receipt.json` (receipt sha
`622e3d115a6232e64cfd355540d6357e385325fc59fb29b3687fc7df1d99e45c`). Because admission failed,
the seal and MAIN fire-order gates did not fire.

## Typed rung and family status

| Rung | Screened body | Fresh publish | Full scorer | Typed disposition |
|---|---:|---|---|---|
| full-set identity | 176,436 B (`-3,756 B`) | `NOT-FIRED`; identity control only | `NOT-FIRED` | `CONTROL-COMPLETE` |
| `tau_1e-8` | 177,856 B (`-2,336 B`) | `NOT-FIRED`; charter selected one byte-real best rung | `NOT-FIRED` | `FOLDED-THIS-ARM`, retained candidate |
| `tau_1e-7` | 177,481 B (`-2,711 B`) | `NOT-FIRED`; charter selected one byte-real best rung | `NOT-FIRED` | `FOLDED-THIS-ARM`, retained candidate |
| `tau_1e-6` | 177,227 B body; 177,252 B published (`-2,940 B`) | `PUBLISHED`, repeat-identical | base + candidate n600 complete | **`INSTANCE-REFUSED-SEG-BAND`** |

The charter's formulation falsifier did **not** fire: a gate-clean pair-screened rung retained
2,940 B, not under 45 B. Pose-screened re-selection is therefore not closed at formulation scope.
What closed is this selected rung as an admissible realized candidate. The two stricter retained
rungs remain untested by fresh compensation and the full scorer; they are not silently promoted or
silently declared negative.

This arm does not fire the registered fcd1 batch ladder and does not claim #1295 closure. MAIN owns
all Modal dispatch; fcd3 made no Modal call.

## Verification

- `ruff check --select E9,F` and `py_compile` pass on
  `experiments/ddm_fcd3_pose_screened_reselection.py`.
- Two genuine `review_tracker.py mark-file` passes are recorded for the new Python file.
- `pytest -q experiments/tests/test_ddm_fcd1_field_for_coder_diagonal.py
  experiments/tests/test_ddm_up3_carrier_splice_entropy_riders.py` passes `6/6`.
- The n600 inverse-coder control is byte-identical, and the full-set pair selector reproduces both
  the union token field and union archive SHA-256 exactly.
- Both 38-stage scorer runs completed under safe-run receipts; their aggregate S values were
  recomputed from components rather than copied from a rounded evaluator printout.
- `upstream/` was read-only. Unrelated shared-worktree and staged-index state was preserved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN assigns the next fcd family arm; consumer store:
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/fcd3_pose_screened_reselection/`;
  fire trigger: MAIN explicitly reopens task #1320 with the scorer lane free. Run the retained
  strict `tau_1e-8` body through the same fresh Schur -> repeat publish -> sequential n600 gates;
  admit only on the realized component band, otherwise record an instance refusal.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN assigns a position-selector successor; consumer store:
  the existing `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/` store under a new
  successor subtree; fire trigger: the remaining pair-level rung is refused or MAIN folds its EV.
  Build the charter-named position-level selector with joint realized Seg/Pose effect, real
  full-stream re-encoding, fresh exact-object compensation, and the same ordered scorer gates.

## LIVE-HYPOTHESES

- A stricter retained pair rung may trade enough diffuse Seg spill for an admissible net while
  preserving material rate credit. This remains plausible because `tau_1e-8` already retains a real
  2,336 B credit, but it is untested under fresh compensation and the full scorer.
- Position-level joint Seg/Pose screening may separate useful coder positions from harmful effects
  hidden inside a kept pair. Pair granularity solved pose and rate but worsened Seg across 404
  pairs; that diffuse pair-level pattern makes another worst-pair drop weak, while leaving
  within-pair selection genuinely untested.

## DEAD-ENDS

- Entropy, average-price, and additive-credit pricing are closed for this ladder. Every rung was
  encoded as a complete real token stream, and the identity rung was checked at archive SHA level.
- Carried union compensation is closed as publish evidence. Retained fcd2 values route the screen
  only; a fresh solve on the exact selected body is mandatory.
- Exact B/H token labels are not realized SegNet flips. The ordered scorer now directly falsifies
  that transfer for published `tau_1e-6`: the archive is smaller and pose-safe, but Seg regresses.
- The published `tau_1e-6` instance is closed for sealing, READY status, and dispatch. Its advisory
  net delta is positive by `0.0019433243907622244`, far beyond the admission band.
- Another pair-level worst-k drop based only on the realized Seg census is closed as a justified
  shortcut: positive Seg harm is diffuse, with the worst ten pairs carrying only 6.75%.
- Widening the carrier is still dominated on this object. Fresh int12 compensation already clears
  the pose gate for only +25 B, so paying the estimated ~3,600 B int16 widening cost has no live
  pose deficit to buy down.
- Scoring, sealing, or dispatching a publish-refused archive remains closed by the compile ordering
  contract; no bypass is permitted.

**Own-vehicle frontier: UNMOVED — S `0.14811799921260607` @ `180,215 B` `[contest-CUDA T4, n600]`, gb1 archive sha `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.**
