# DDM JO1 — joint semantic-token / HP3 probability object

**Status:** `QUEUED-WITH-A-FIRE-ORDER` for one MAIN-owned contest-CUDA T4 row. The scorer-free JO1
work is complete. It produced four actual HP3/RC64 archive recloses, one receiver-closed composed
archive with an independent byte-identical repeat, and a sealed exact-row recipe. No scorer was run,
no exact score was measured for the composed archive, and the frontier did not move.

## Measured complete-container result

All rows keep CP135's own HP3 model, semantic renderer, pose carrier, residual table, and runtime.
Only the semantic-token object changes. Bytes are full stored `archive.zip` bytes, not packet sums.

| Direct event object | Events / sites | RC64 token B | archive B | delta vs CP135 | exact rate delta S | disposition |
|---|---:|---:|---:|---:|---:|---|
| calibrated pose-nonspending, one event per sampled pair | 6 / 12 | 115,232 | **186,253** | **+1 B** | +0.000000666 | composed pre-candidate; exact pending |
| EC1 whole alphabet | 200 / 234 | 115,234 | **186,255** | **+3 B** | +0.000001998 | rate mechanism proven; pose diagnostics forbid promotion |
| JS7 pose-gate rejects | 20 / 20 | 115,234 | **186,255** | **+3 B** | +0.000001998 | diagnostic only |
| JS7 selected stack | 44 / 58 | 115,234 | **186,255** | **+3 B** | +0.000001998 | exact-output-equivalent rate reprice; still dead |

CP135 is 186,252 B. The measured result is the DERIVED-site form EC2 left open: the event sites are
not transmitted as a second coordinate section; they are the changed symbols in the receiver's
actual token sequence and are priced by the actual HP3 probabilities and RC64 stream. This is not a
claim that video-derived coordinates are free. It is a different object in which the decoder recovers
the changed sites by decoding the semantic tokens themselves.

The 44-event direct object has raw semantic-token SHA-256
`a78bb2992b3e711b602909ca90ca72dc98c1ab8f6cfcea30594d2f18c53810e0`, identical to the
retained token object rendered by the exact JS7 overlay row. The model, carrier, and residual objects
are also identical to CP135. Therefore JS7's exact `d_seg=0.00029675` and `d_pose=9.06e-6` transfer by
same rendered object, while rate changes from 186,575 B to 186,255 B. The derived reprice is
**S=0.16321296254120266**, 0.000213075 below the explicit-overlay row but still about 0.001258 worse
than CP135. This is a component-identity derivation, not a new `upstream/evaluate.py` row.

## Composed object and score gate

The composed archive is:

- `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/retained/candidates/calibrated_pose_nonspending/primary/objects/archive.zip`
- 186,253 B; SHA-256 `cbcbb9ec22f81ad6ce2f8f97c976148831e825ba64312878a798d92a46907c8f`
- adapted runtime:
  `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/retained/candidates/calibrated_pose_nonspending/primary/adapted_runtime/`
- independent RC64/archive repeat:
  `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/retained/candidates/calibrated_pose_nonspending/determinism_repeat/objects/archive.zip`,
  byte-identical with the same SHA-256.

The six events are the best `B/robust-flip` singleton under three conservative restrictions: existing
JS7 pose gate, non-positive singleton pose delta, and at most one event per sampled pair. Their n32
singleton diagnostics sum to -241 projected robust flips and -7.508e-6 pose delta. Those sums are
**not composition authority**: JS7 already measured a sign reversal at n600, and a pose sum below the
physical zero floor is itself evidence of non-additivity. Even the deliberately optimistic, non-authority
scenario that sets pose to zero and transfers the discredited -241-flip sum gives S=0.1534569294, not
sub-0.15. This is a prefilter observation, not a lower bound and not a score projection.

The frame-0 Schur antidote was not invoked because none of the six selected singleton events spends
pose on the existing n32 measurement. It becomes mandatory before promotion if the n600 row reports
positive pose debt. The sealed MAIN recipe is
`.omx/research/ddm_jo1_joint_probability_object_20260812_t4_recipe.json`; it pins the archive, repeat,
runtime, claim command, single CUDA command, recovery command, owner, consumer store, and fire trigger.

## Bending-energy feature verdict

**REJECT for event ordering.** The implemented feature reconstructs the affected source/target lattice
contours, splits them into contour components, and measures the local discrete turning-energy change.
Under leave-one-pair-out top-5 evaluation inside the existing pose gate:

- bending ordering: 502 held-out robust-flip gain, 15/30 useful selections;
- existing `B/flip`: 874 held-out robust-flip gain, 30/30 useful selections;
- the fitted direction was stable across held-out folds, but the preferred direction disagreed between
  boundary and lane strata.

It fails both DG1 falsifiers: lower held-out useful yield and unstable sign by stratum. It is not in the
composed object's ordering and should not be retried on this 200-event store.

## Receiver closure, retention, and reproducibility

For each object JO1 retained the edited 117,964,800-symbol spatial plane, its F26 event-order array,
event application ledger, source manifest, every probability checkpoint, 25 RC64 checkpoints, RC64
payload, two full decoded-symbol payloads, physical sections, member `p`, archive, deterministic ZIP
repeat, and shipped-backend parse-back. Changed frame `f` and `f+1` probabilities were recomputed with
the real F26/HP3 teacher-forced model; unaffected probabilities were byte-identical hardlinks with
path-local custody receipts. All 600 frames were then freshly RC64-coded and decoded.

The encoder verifier backend and shipped receiver backend are different source files, so JO1 did not
infer equivalence from names. It compiled the shipped backend and independently decoded all 117,964,800
symbols for every object. Event-order and spatial digests matched the intended retained payloads. The
CP135 HP3, semantic, pose-carrier, and residual sections parsed byte-identically in every archive.

The governed run completed in 336.28 s with exit 0. The sandbox denied `ps`, so safe-run's reported
0 MiB RSS is **not a memory measurement**. The 4 GiB kill cap and 8 GiB storage admission were still
declared. Retained custody is 4.6 GiB under `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/`; no
partial files remain and no materialized payload was deleted. The first attempt stopped before a new
probability frame because copied receipts named their source paths. JO1 regenerated only local receipt
metadata and resumed from the retained event/token payloads.

Durable receipts:

- final table: `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/FINAL_RESULT.json`, SHA-256
  `eee5c58d4279a122e9c1c0bd87e2dc64d78774b26efca2675688fc436d903315`
- analysis and bending gate: `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json`,
  SHA-256 `440542cfc34e7bbd8ff2e2a1fd71e4f62aea1f929f1229c4aae5c68e29323a3c`
- governed-run receipt: `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/run/safe_run.json`,
  SHA-256 `1637f526c02ca8548a5cc7b82dd0f50b03ec29e52670ba21cbaed186b84e9e6e`
- runner source SHA-256 `77990654e90f4c4cd0d2b068e0f039da171889d0ecc6ec418ac5900868d688a9`
- focused test source SHA-256 `ee38eb8d32b0b7dc6ae735755baa633a3038a72fbeb928e550223d102f0c8892`
- T4 recipe SHA-256 `4d816b9f08205dd3a031ac4041daaa85a8e49c271156ab60a5ac750039d4de89`

## RECALL EVIDENCE

Before design, `tools/corpus_query.py` searched all seven durable stores: research 8,459, equations
886, memory 2,114, DAG 915, council 297, tasks 531, and docs 96. Exact query strings were:

- `joint probability object HP3 RC64 event alphabet complete container`
- `bending energy elastica event ranking B per flip`
- `frame0 Schur pose antidote calibrated pose stack`
- `derived site event coordinate implicit conditioning complete to complete`

Direct content searches also covered `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG/FEED, task-ledger
surfaces, `main_hot_state.md`, and the active lane registry for `jo1`, `probability object`, `event
alphabet`, `HP3`, `RC64`, `bending`, `Schur`, and `complete-to-complete`. No separate JO1 task row or
scorer ownership was found in those bounded index/DAG/task/registry scopes; the committed charter and
the hot-state ps135-to-jo1 chain are the governing rows.

The canonical equation registry was inspected directly. Decision-relevant entries were
`score_marginal_lagrange_multipliers_v1`, `categorical_blahut_arimoto_rate_distortion_v1`,
`scorer_conditional_joint_rate_distortion_floor_v1`,
`pr95_family_l30_range_arithmetic_coding_categorical_v1`,
`indirect_rd_logloss_equals_information_bottleneck_v1`, `seg_rate_breakeven_v1`,
`worldsheet_transport_residual_event_rate_v1`, and `ddm_lp1_deepest_home_context_waterfill_v1`.

Recall beyond the charter seeds changed the implementation in five ways:

1. T1R1 proved the exact CP135 exporter/RC64 receiver path could consume an arbitrary retained token
   plane. JO1 reused that physical mechanism while keeping CP135's carrier, avoiding the refuted ps135
   carrier.
2. SR1 showed the shipping prior already includes strong edge/previous-frame context. JO1 therefore
   changed the source symbols and recomputed the existing probability object instead of adding another
   generic edge table.
3. EC2's corrected +213 B complete-package loss and HP3's nonlinear section evidence forced actual
   complete archive races. No EC1 coordinate bytes or token estimates were added algebraically.
4. HC1 proved probability carriage can be exact while realization is catastrophic. JO1 therefore
   proves token/receiver closure but leaves distortion to n600 authority.
5. JS7's exact sign reversal and pose-stack derivative made n32 sums non-admissible, produced the
   1.3e-7-class stack budget, and prevented an unearned exact-score claim.

## Validation

- `pytest -q experiments/tests/test_ddm_jo1_joint_probability_object.py`: 5 passed, including the
  targeted P0 payload-retention gate over both JO1 Python files.
- Ruff and Python compilation: passed.
- P0 static retention census: 2/2 Python files examined, one detector-candidate file parsed, zero
  measure-and-discard findings.
- Review tracker: full rescan indexed 29 runner entities and 5 test entities; two post-fix
  `mark-file --status reviewed` passes recorded at 100% coverage for both files.
- All four fresh RC64 decodes and all four shipped-backend decodes: 117,964,800/117,964,800 symbols
  exact per object.
- Composed independent RC64 and archive repeats: byte-identical.
- Exact evaluator: not run by JO1; queued to MAIN in the sealed recipe.

The fast developer preflight reported 17/25 green and eight codebase-wide red gates:
strict-load state writers, authoritative-tag custody metadata, codebase drift, dispatch-claim helper,
landing solver wire-in, lane preregistration, substrate scorer contracts, and substrate pose defaults.
This is not a codebase-wide green claim; those gates were not weakened or waived for JO1.

## Boundaries and disposition

- **MEASURED:** all token/archive bytes, hashes, complete-container deltas, event/site denominators,
  probability/decoder closure, bending held-out result, and deterministic repeat.
- **DERIVED from exact same rendered object:** the 44-event direct reprice S=0.16321296254120266.
- **NOT MEASURED:** composed n600 d_seg, d_pose, exact S, or any contest score for archive `cbcbb9ec…`.
- **QUEUED-WITH-A-FIRE-ORDER:** owner MAIN exact contest-CUDA scorer owner; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/main_t4`; fire when MAIN owns the sole scorer lane,
  verifies archive/runtime pins, and accepts the recorded non-authority prefilter risk.

Effective frontier remains **CP135 S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.
Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.
