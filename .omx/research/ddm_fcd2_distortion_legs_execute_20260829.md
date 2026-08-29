# ddm_fcd2 distortion legs execution — fresh union compensation refuses the publish gate

**Date:** 2026-08-29  
**Task:** #1319  
**Owning charter:** `.omx/research/charters/ddm_fcd2_distortion_legs_execute_20260829.md`  
**Status:** `INSTANCE-REFUSED-POSE-GATE`; full frozen-scorer pair, seals, and Modal dispatch `NOT-FIRED`; family `OPEN`  
**Axis:** `[macOS-CPU frozen-scorer advisory, DALI-lineage GT]`; `score_claim=false`; `promotable=false`

## Result

The union did not reach the scorer gate. A fresh candidate-bound n600 Gauss-Newton solve, a full
diminishing-returns refinement, and two byte-identical closes reduced the union's uncompensated
`d_pose` from `0.0016055422933954212` to `0.00027348054805362656`. The same-instrument jt21 base is
`0.0000063656845167356244`. The in-compile requirement was

`d_pose_after <= d_pose_base + 1e-8`.

It failed by `0.0002671048635368909`, or `26,710.49x` the declared MSE band. The retained terminal
receipt is `REFUSED_POSE_GATE`. No published runtime was created. Because publication is the explicit
precondition for the two full n600 scorers, I did not run them, did not make a seal, and did not ask
MAIN to dispatch Modal.

This is an **INSTANCE** verdict on the exact fcd1 union plus the current 12-dimensional carrier solve.
It is not a realized SegNet verdict, a net-score measurement, or a family closure.

## RECALL EVIDENCE

I searched the full research corpus by content for `field-for-coder`, `pose-free`, `schur`, `hpac`,
`token field`, `marginal re-encode`, and `real re-encode`; inspected the canonical research indexes,
`sub015_DAG_*` FEED blocks, design/spec surfaces, canonical task-status and harness-bridge stores; and
ran `tools/list_canonical_equations.py --json`.

Beyond the charter's named seeds:

- `ddm_sa3_compensated_edit_rebased_verdict_20260818.md` reinforced that compensation must be on the
  exact edited object and that rounded two-row bounds cannot be added as if they were one row.
- `ddm_fs3_jg5_real_price_reopen_20260820.md` reinforced real marginal re-encode pricing, not entropy
  estimates.
- `ddm_jg5_pose_resolve_on_edited_renders_20260819.md` supplied the proven disjoint `5 x 120` full-n600
  sharding pattern. I used that pattern instead of a monolithic 600-pair process.
- Relevant registered laws were `score_marginal_lagrange_multipliers_v1`,
  `pairset_component_marginal_score_decomposition_v1`, `token_rate_model_direction_dependence_v1`,
  `greedy_set_average_vs_marginal_price_v1`, `hpac_mc36_joint_descent_law_v1`, compensated semantic
  edit, and `section_coding_axis_closure_v1`.
- I did not find task #1295 or #1319 rows in the exact scopes
  `.omx/state/canonical_task_status.jsonl` and
  `.omx/state/harness_tasklist_bridge_20260803.jsonl`; the charter and fcd1 memo therefore remained
  the route authority. This is bounded absence in those two stores, not a claim of global absence.

The recall changed execution in one material way: both the GN and refinement populations were five
disjoint 120-pair shards, with all 600 rows required before adjudication. It did not change the
candidate, receiver, score band, or publish gate.

## Trigger, custody, and lane

- qbt2b r10 was terminal `rc=0` at `2026-08-29T16:50:55Z`; its claim explicitly passed the scorer
  surfaces to fcd2.
- fcd2 claimed `ddm_fcd2_scorer_20260829` at `2026-08-29T16:57:19Z` and terminated it at
  `2026-08-29T18:17:54Z` as `refused_pose_gate_no_full_n600_scorer`.
- All materialized payloads and receipts remain under the existing consumer store
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`.
- The exact base archive is `180,192 B`, sha
  `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3`.
- The exact fcd1 union body is `176,436 B`, sha
  `c45ab4e687d1a598b2c2191e5c4bf176bb1c12b24748795434cd109eb9a3aa6b`.
- The retained base raw is `3,662,409,600 B`, sha
  `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`;
  the union raw is the same size, sha
  `042fad94690563d774f9480a0fd136334f7c91e4867386587655d15dd04dff19`.
- Storage stayed on APDataStore as the charter required. Preflight observed about 30 GiB free before
  the solve; no re-decode or duplicate 3.66 GB scorer worktree was needed.

## Apparatus refusal and permanent repair

The first close correctly failed before emitting an archive. The inherited up3 splice parser tried
to read the live jt21 carrier as plain Rice and raised `Rice value exceeds int12`. The live body has
RX1 reserved value `0x1a`: RR5 arithmetic-basis and DX2 CABAC-coefficient riders are both present.
The parser restored neither before reading CAP1.

I repaired `experiments/ddm_up3_carrier_splice.py` to follow the public receiver's exact inverse
order, RR5 then DX2, and to apply the forward order, DX2 then RR5, when rebuilding. Both the base and
union now rebuild byte-identically at their original archive SHA-256 values. A focused regression
test pins the forward/inverse order and the no-rider identity path. I also repaired
`experiments/ddm_fcd1_incompile_schur.py` so a measured publish refusal is written before it raises;
the former exception-shaped absence would have discarded the terminal gate receipt.

The failed close receipt remains retained at
`fcd2_distortion_legs/union/schur/close/safe_run_status.json` (`2,592 B`, sha
`5d749074b3dd8e821bb0f85ba5a17400284dfe18333f3599c67b45b6f8c43835`). It is excluded from all
closure arithmetic.

## Fresh solve

The baseline receipt is
`fcd2_distortion_legs/union/schur/baseline/BASELINE.json` (`2,509 B`, sha
`188f832c4ca3476e9372bc5bd1a43d0dd7f5458481bdec443ffd1ea0a1fd0913`). It bound the exact archive
and both retained raws, measured all 600 pairs, and found 547 pairs worse and 8 better on the union
before compensation.

One initial monolithic GN attempt reached 25/600 rows and was stopped. Its 25 rows and terminal
`killed`, exit-130 receipt remain under `union/schur/gn/`; they are excluded. The admitted solve was
five disjoint shards, each 120/120 with exit 0: together 600 unique pair ids spanning 0..599 with no
duplicate or missing pair.

The default close reached `d_pose=0.0002762680435646151`. To avoid treating a budget-stopped solve as
optimal form, I then ran the tool's full diminishing-returns refinement on all 600 pairs, again as
five disjoint 120-pair shards. All shards exited 0. Stop reasons were:

| Stop reason | Pairs |
|---|---:|
| `no_improving_step` | 597 |
| `converged_below_materiality_floor` | 2 |
| `lattice_floor` | 1 |
| any GN/outer budget stop | 0 |

Selecting the lower measured pose value per pair across the initial and refined banks produced the
final `d_pose=0.00027348054805362656`. Thus refinement changed the mean by only
`-0.00000278749551098854` and did not approach the publish gate.

## Schur publish receipt

| Field | Measured value |
|---|---:|
| same-instrument jt21 `d_pose_base` | `0.0000063656845167356244` |
| uncompensated union `d_pose` | `0.0016055422933954212` |
| final candidate-bound `d_pose_after` | `0.00027348054805362656` |
| `d_pose_after / d_pose_base` | `42.96168736207959x` |
| pose MSE band | `0.00000001` |
| excess above `base + band` | `0.0002671048635368909` |
| repeat archive | `176,463 B`, sha `d4f6b9321b9ede31d39417fd33c601be64cd7b3603d41cc4374e688197c4c4a3` |
| repeat identity | archive bytes identical; `d_pose` identical |
| frame-1-producing sections | HPAC, semantic, and tail byte-identical to the union body |
| publish verdict | `REFUSED_POSE_GATE` |

The two close receipts are:

- `close_refined/CLOSE.json`: `10,966 B`, sha
  `b8275253226a5d8990f694bc7b189bc58b76b58ee2ad9258629a073e8970aeaa`;
- `close_refined_repeat/CLOSE.json`: `11,015 B`, sha
  `514f8b933ee634de881fa813c75ebf0c57c173331d3afd810b6e5e8dfa02ed3b`.

The terminal publish receipt is
`publish_refined/PUBLISH.json`: `2,600 B`, sha
`f849374f949e55d564337d41f47a415bf7b9e993d73c790d2b4926a3491c044d`.
`published` is `null`, and the requested destination runtime is absent.

## Full scorer table and score arithmetic

The charter only allows the full frozen scorers after a repeat-identical publish. That condition
failed, so the honest table is:

| Body | Archive bytes | realized n600 `d_seg` | realized n600 `d_pose` | S recomputed from components | Disposition |
|---|---:|---:|---:|---:|---|
| jt21 base | 180,192 | `NOT-FIRED` | `NOT-FIRED` | `NOT-COMPUTABLE` | control retained; scorer precondition failed upstream |
| compensated union | 176,463 | `NOT-FIRED` | publish-instrument `0.00027348054805362656`; full evaluator `NOT-FIRED` | `NOT-COMPUTABLE` | `INSTANCE-REFUSED-POSE-GATE` |

The exact final rate leg against the jt21 base is `-3,729 B`, hence

`Delta S_rate = 25 * (-3729) / 37,545,489 = -0.002482988036192577`.

The measured publish-instrument pose difference contributes

`Delta S_pose = sqrt(10 * 0.00027348054805362656) - sqrt(10 * 0.0000063656845167356244)`
`= +0.04431684368081841`.

Thus the two known legs sum to `+0.041833855644625835`, but this is **not net Delta S**: the realized
SegNet leg was not authorized to run. I do not report an advisory score, an admit-band net verdict,
or the charter's projected `~0.1456` as a measurement.

## Folded orders and verdict scope

| Row | Disposition | Reason |
|---|---|---|
| union | `INSTANCE-REFUSED-POSE-GATE` | candidate-bound, repeat-identical n600 solve remains 42.96x base pose MSE |
| batch0 | `FOLDED-NOT-FIRED` | registered trigger requires a published union with realized refusal `<=5x`; publication failed before realized net scoring |
| batch2 | `FOLDED-NOT-FIRED` | same trigger; no inference from union to batch |
| batch1 | `FOLDED-NOT-FIRED` | same trigger; no inference from union to batch |
| dual-axis seals | `NOT-FIRED` | no locally admitted published candidate |
| MAIN Modal order | `NOT-FIRED` | MAIN retains dispatch; no sealed candidate exists |
| #1295 family closure | `NOT-CLAIMED` | requires refusal `>5x` across at least three independent batches; zero batches were scored |

The current evidence closes only the exact union/current-carrier instance. It does not close
pose-aware re-selection, a richer carrier basis, or the three independent batches.

## Verification

- `ruff check --select F,E9` passed on both repaired modules and the regression test.
- `py_compile` passed on all three Python files.
- `pytest -q experiments/tests/test_ddm_fcd1_field_for_coder_diagonal.py experiments/tests/test_ddm_up3_carrier_splice_entropy_riders.py`: `6 passed`.
- Two clean `review_tracker.py mark-file --status reviewed` passes were recorded for each changed
  Python file.
- Live-object identity controls rebuilt both base and union at their original SHA-256 and byte size.
- `upstream/` was not modified. No Modal call was made. Unrelated shared-worktree changes were not
  staged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store:
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: a new
  candidate-bound carrier or solve on the exact union archive produces two byte-identical n600
  closes with `d_pose_after <= 0.0000063656845167356244 + 1e-8`. Run `publish`, then the base and
  union full frozen scorers sequentially; recompute S from components and seal only beyond the
  `+/-3.5e-6` band.
- **FOLDED** — owner: MAIN; consumer store:
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: the preceding action
  yields a published union whose realized net refusal is `<=5x` the canonical band. Process batch0,
  batch2, then batch1 independently through decode/Schur/publish/full-scorer gates.
- **FOLDED** — owner: MAIN; consumer store:
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: at least three
  independent re-selection batches each produce realized refusal `>5x`. Only then write the #1295
  FAMILY closure; do not promote this union instance into that claim.

## LIVE-HYPOTHESES

- A richer candidate-bound pose carrier may clear the hard tail. The current solve demanded up to
  about 90,109 code units, reached the signed-int12 boundary, and left 597/600 refinement rows at
  `no_improving_step`; that is the shape of carrier under-capacity, not evidence that all legal
  compensation is impossible.
- The three disjoint field batches may be pose-safe even though their union is not. Each has about
  one-third the edit density, and scorer/carrier costs are known not to compose additively; this is
  plausible but deliberately untested because the registered folded trigger did not fire.
- Pose-aware token re-selection may retain part of the field-coder byte credit while avoiding edits
  transverse to the carrier tangent. The current fcd1 selection used exact B/H coding labels but no
  candidate-bound pose Jacobian, and 547/600 pairs worsened before compensation.

## DEAD-ENDS

- The exact fcd1 union with the current 12-dimensional fresh GN plus diminishing-returns solve is
  closed at INSTANCE scope: repeat-identical `d_pose` remains 42.96x the base control.
- Six-iteration GN alone is not an admissible rescue claim. Full refinement improved the population
  mean by only `2.7875e-6`, with no GN or outer-round budget stop.
- Parsing RR5/DX2 bytes as plain Rice is closed as an apparatus defect. The forward/inverse rider
  path now has exact base and union identity controls plus a regression test.
- Carrying qs2/qs4/qs5 compensation, using entropy/average prices, adding separately measured
  credits, or calling exact B/H token labels realized SegNet flips remain closed by the inherited
  negative evidence and were not retried.
- Running full scorers, making seals, or dispatching Modal from a publish-refused archive is closed
  by the charter's receiver/publish ordering and was not bypassed.

**Own-vehicle frontier: UNMOVED — S `0.14811799921260607` @ `180,215 B` `[contest-CUDA T4, n600]`, gb1 archive sha `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.**
