# G118 adversarial review — G111 authority composition

Date: 2026-07-27T13:40:23Z  
Lane: `lane_g118_g111_composition_adversarial_review_20260727`  
Mode: bounded read-only architecture and code review plus focused local tests  
Authority: `[macOS-CPU structural verification; non-promotable]`  
Truth: `research_only=true`; `candidate_claim=false`; `score_claim=false`;
`promotion_eligible=false`; `pointer_moved=false`

## Outcome first

The G111/G112 physical producer-lineage machinery is not the missing
composition layer. It requires the cold root, paired deploy/resume checkpoints,
recursive lineage chain, target projection, semantic child, and generated-Y1
pose initializer to exist as physical objects and reopens them rather than
trusting names. This review did not claim that a real trained G111 stage already
exists.

The immediate gap is one layer later: committed G117 is deliberately an
**engine with untrusted injected inputs**, not the production authority wrapper
that can convert a physical G112 stage into an exact parsed-public-wire
observation. That honest boundary is the P0 gate. The wrapper must derive
labels, scorer, checkpoint identity, pose initializer, runtime identity, and
the live target from reopened custody; it must not accept caller assertions for
them.

The second P0 is cross-stage survival. The trainer still promotes a single
`BEST` solely by floating-state realized `d_seg`, while G117 advertises a
heuristic “nondominated and semantically second-best” retention policy. Neither
is sufficient for the conditional codec. Before pose refit, the only
nonarbitrary safe pruning theorem is the exact nonnegative-term obstruction:

```text
retain stage i  iff  100*d_seg_public_wire(i) < T_live
prune stage i   iff  100*d_seg_public_wire(i) >= T_live
```

Every retained stage remains alive until its own post-G105 conditional
`Y0 | Y1` solve and same-object final archive pricing. A rank-zero semantic
archive action at or above the frontier is **DEFER**, not a kill, because the
final conditional packet and outer compression are not known to be monotone in
the rank-zero ZIP bytes. This is the missing micro-to-macro bridge: preserve
every semantic state not mathematically excluded by the full score, then let
the conditional pose value function and exact ZIP race decide.

No exact archive, n600 evaluator row, candidate, or frontier movement was
produced by this review. The canonical pointer observed during review remained
the external official-leaderboard display target `0.172`; all production logic
must reload it dynamically rather than copy that literal.

During memo landing, the parent adopted these findings into committed G120 and
G121 implementation contracts. That closes design ambiguity, not execution:
the committed G120/G121 artifacts are specifications, and the G120 Python/test
files observed afterward were still untracked sibling-owned work in progress.
This review did not absorb or confer authority on them.

## Frozen review snapshot

Committed stack observed during review and pre-commit reconciliation:

| object | identity |
|---|---|
| reviewed G117 base | `dbca26e2c76c7ef4e70a4a6f02b7a9eb128e1450` |
| pre-commit shared-tree HEAD | `4d9418e0bc` (`G111: bind launch gate to external exhaustive stage compiler`) |
| G120 authority contract | `ebed6dd1fc`, amended by `31b1b3f773` |
| G121 exhaustive-harvest contract | `f5cfb242ff` |
| G111 external-harvest launch gate | `4d9418e0bc` |
| G111 governed-launch routing | `b144d655ae` |
| G111 held-producer contract | `cdecd8130e` |
| G110 landing | `2403b8093db21f1d376be5a8ef5117c730111f8b` |
| G111 landing | `a0164871e103a128af1e32ef6cabc34056944ccb` |
| V10 landing | `239554cecb2573d4a8654e8fc5fef12383c2e8be` |
| G105 landing | `51e319d862858ee55acad65da65910670c176ee8` |
| G109 landing | `c0d36b4e99ff20f8e05c2ef65ad341130f7b767a` |
| G115 landing | `fd48588a5f33ed0ac3b3b9079e8e5406766bf492` |
| current G117 source SHA-256 | `6956d8e43db7aca5de428a6cace236b8a5c3ca8da8de5104f2a2c3f1a54e5f92` |
| current G110 compiler source SHA-256 | `cd863bd4f2dcae5e4b61912efd76f86bb4ee22fb2f1a56e5be9003e6bbbe3f00` |
| current public G110 `inflate.py` SHA-256 | `1daf6e5862524399a5a19426cf82f092a9c7a9096d66dcaa603936a402d57147` |
| current G112 source SHA-256 | `4133a841baa82f882507cc8d8d790d1e31d23f0c4f5083f55df4b41092438ae4` |
| current G109 capsule source SHA-256 | `cc9e660694eace117eba3efb4bb086248cd3e44d94dcdd39d7c64bf3793132bd` |
| current dynamic-pointer helper SHA-256 | `1af0f115d957745f53932b3dadcaa64d053bd12ae25fd20b1d902fb1de262487` |
| current G115 source SHA-256 | `46d3c1a3bcc4e8d84cbcf4bd4eb61da16ba1ecca41339ebf1923716617500868` |
| current trainer source SHA-256 | `7befc939950e2825f1485b6a73d89b0688dcc554a75b90d3584c458cdae88f2a` |
| observed canonical pointer SHA-256 | `06027fa2dcb695d755858d6e101c7be57f1fbe5570903eb4bf8c5e2f38510388` |

`AGENTS.md` and `CLAUDE.md` reopened byte-identical at SHA-256
`8dae16d9ef854a9729f8979ba1695761aad489546d7a0722b3325d71b39a6526`.
The worktree was shared and dirty. This lane did not edit or attribute any
pre-existing file.

## P0-1 — production authority wrapper is required

Status at pre-commit reconciliation:
`CONTRACT_LANDED_IMPLEMENTATION_SIBLING_WIP_NOT_YET_AUTHORITY`.

### Evidence

Committed G117 truthfully fences itself:

- `src/tac/witness_dsl/g111_parsed_g105_stage_selector_v1.py:948-966`
  accepts caller-provided config, parameters, Y1, labels, scorer callback,
  scorer/checkpoint/initializer hashes, target float, pointer hash, and stage
  tag.
- `:981-984` requires `injected_inputs_are_test_only=True` and refuses the
  production surface.
- `:1190-1198` records `engine_only=true`,
  `production_authority_closed=false`,
  `production_wrapper_required=true`, and no production verdict or admission.

That is honest engine code, not a defect. Treating its receipt as production
authority would be a NO-FAKE authority upgrade.

### Smallest correct API

The production entrypoint should expose only physical roots and durable output
locations:

```python
compile_select_g111_stage_production_v1(
    *,
    g112_partition_receipt: Path,
    expected_g112_partition_receipt_sha256: str,
    repo_root: Path,
    out_dir: Path,
    progress_dir: Path,
) -> G111ParsedG105ProductionSelectionV1
```

It must not accept target labels, a scorer callback, scorer identity, source
checkpoint identity, pose initializer identity, target score, pointer identity,
or stage tag from the caller. Those are derived facts.

### Required internal derivation

1. Call
   `open_g112_partition_receipt(...)`
   (`taskspace_g112_exact_checkpoint_partition_v1.py:903-1075`). This
   recursively proves the current G111 deploy/resume pair and complete fresh
   lineage, then reopens the exact semantic child and conditional initializer.
2. Derive the physical G109 aggregate receipt and expected hash from the
   checkpoint's recomputed target projection. Open it with
   `V9TrainingTargetCapsuleLoaderV1.open(...)`
   (`taskspace_v9_training_target_capsule_v1.py:1099-1156`). Its production
   preflight already rehashes G46 labels, source video, SegNet/PoseNet weights,
   reviewed upstream closure, package/runtime custody, and batch-16 geometry
   (`:430-530`).
3. Construct the frozen CPU SegNet internally from that reopened upstream root
   and weight file. Derive scorer identity from the G109 receipt, weight hash,
   upstream closure, Python/package custody, wrapper source, exact V10
   realization, and `37x16 + 1x8` execution geometry. An arbitrary callback SHA
   is not authority.
4. Call `load_dynamic_frontier_target(...)` before measurement and
   `verify_dynamic_frontier_target_snapshot(...)` after all 600 pairs
   (`dynamic_frontier_target.py:214-320`). Never accept an independently
   supplied float plus hash.
5. Invoke a private measurement core shared with the existing test-only engine.
   Do not relabel or mutate an engine-only receipt into a production receipt.
6. Derive the stage tag from the physical checkpoint ID/stage and emit a new
   production schema binding all reopened objects.

The receipt must bind the actual G112 receipt path/SHA, deploy checkpoint
path/SHA, full-state resume checkpoint path/SHA, lineage receipt path/SHA,
checkpoint ID, semantic child, pose initializer, selected G105 packet, selected
G110 rank-zero packet/archive, scorer/runtime identity, and pre/post pointer
snapshot.

Current G117 `paired_resume_identity`
(`g111_parsed_g105_stage_selector_v1.py:912-916`) binds only its batch progress
key and batch-receipt chain. Those are scorer-resume artifacts, not the paired
trainer deploy/resume checkpoint. Production must not reuse that misleading
field shape.

## P0-2 — enforce exact-obstruction retention in the controller

Status at pre-commit reconciliation:
`RULE_ADOPTED_BY_G121_SPEC_CONTROLLER_IMPLEMENTATION_OWED`.

### Current sharp edge

The trainer's `_is_new_best` admits only a strictly lower floating-state
realized `d_seg`
(`experiments/train_levelset_witness_realized_through_R_mlx.py:1258-1262`).
`_maybe_preserve_best` then overwrites
`levelset_witness_ema_BEST.npz` and `levelset_best.json`
(`:10855-10895`). This can orphan a stage whose slightly worse semantic
distortion yields a much cheaper conditional pose repair or a better final ZIP.

G117's cross-stage row states
`retain_nondominated_and_semantically_second_best_stage_rows`
(`g111_parsed_g105_stage_selector_v1.py:917`). “Second-best” is arbitrary, and
semantic dominance is not full Seg/Pose/rate dominance while the conditional
value function is unmeasured.

Committed G121 now assigns exhaustive immutable-stage enumeration to an
external resumable controller and states the exact retain-all-below-obstruction
rule. That is the correct architectural response. At this snapshot it remains
an implementation contract, so the legacy trainer/G117 surfaces must not be
mistaken for the controller.

### Exact rule

For stage `i`, let:

```text
N      = 600 * 384 * 512
k_i    = exact public-wire disagreement-pixel count
d_i    = k_i / N
T      = dynamically reopened effective frontier
S_i    = 100*d_i + sqrt(10*d_pose_i) + 25*B_i/37_545_489
```

Because both remaining terms are nonnegative:

```text
100*d_i >= T  =>  S_i >= T for every future pose solution and archive size.
```

This is an exact stage-specific obstruction. Its converse does not hold.
Therefore:

```text
if 100*d_i < T:
    RETAIN_UNTIL_POST_G105_CONDITIONAL_REFIT
else:
    PRUNE_EXACT_DISTORTION_OBSTRUCTION
```

Implement the comparison with the integer disagreement count and the exact
decimal/rational pointer value, not a binary-float epsilon:

```text
100 * k_i < T * N
```

At the observed display target `T=0.172`, this illustration becomes
`d_seg < 0.00172`, or at most `202,899` disagreement pixels out of
`117,964,800`. The production rule still reads the pointer; neither number is a
launch literal.

If the pointer refreshes, reclassify already measured rows from the preserved
integer counts; do not rerun SegNet solely because the target changed.

Do **not** prune when the rank-zero semantic action

```text
A_sem_i = 100*d_i + 25*B_rank0_i/37_545_489
```

is at or above `T` while `100*d_i < T`. Record
`DEFER_POST_G105_CONDITIONAL_REFIT`. The final conditional bytes and DEFLATE
result are an unmeasured joint object, so rank-zero archive size is not a proof
of impossibility.

After the conditional solve, retain the same-object Pareto set over exact
`d_seg`, exact `d_pose`, and final archive bytes, and select only by the full
score. The old `BEST` may remain a training convenience, but it cannot own
candidate-stage survival or the production execution pointer.

“Retain” means every evaluated immutable stage keeps its exact deploy/resume
pair, G112 partition, lineage receipt, and measurement row under SSD custody.
Use hard links or content-addressed references rather than duplicate checkpoint
copies. Removing a row from the active frontier set does not authorize deleting
its bytes without the repository's certify-or-block provenance and cold-store
rules.

## P0-3 — score the exact shipped decoder, then close the public entrypoint

Status at pre-commit reconciliation:
`RULE_ADOPTED_BY_G120_SPEC_IMPLEMENTATION_NOT_YET_COMMITTED_OR_AUTHORITY`.

G117 currently renders with repository functions:

- direct G105 `render_scorer_y1` at
  `g111_parsed_g105_stage_selector_v1.py:652`;
- repository G110 rank-zero helper at `:660-672`.

The shipped receiver separately discovers, imports, parses, and renders exact
plugins in
`submissions/robust_current/g110_two_layer_receiver/inflate.py:207-280`.
Repository-twin equality is not public-wire authority.

The production screen must load the exact sealed runtime/plugin tree that will
ship, render its n600 scorer-Y1 population, and bind that runtime-tree identity.
It should fail unless repository and shipped-runtime population hashes agree.
The G120 contract now requires this exact operation. A sibling-owned
`g120_parsed_stage_production_authority_v1.py` and focused test file appeared
untracked during memo landing; this review did not run, edit, commit, or certify
that moving implementation.
The G110 receipt itself records that clean-extract double decode and contest
timing have not yet run
(`g110_generic_two_layer_public_product_receipt_20260727.json:51,155,173`).

Before any candidate or score claim, the selected final conditional archive
still owes:

1. exact archive bytes and SHA;
2. sealed public runtime tree identity;
3. clean-root parse-back and two independent inflates;
4. bit-identical required output video/raw proof;
5. one full-n600 `upstream/evaluate.py` row on a valid contest-CPU and/or
   contest-CUDA authority axis; and
6. component distances, runtime, source/runtime closure, and pointer promotion
   receipt for those exact bytes.

## P1-1 — remove redundant n600 rendering before the real screen

The current composition performs about three semantic population passes per
wire family:

1. G110 rank-zero construction opens the provider and computes its final-Y1
   binding (`taskspace_g110_generic_two_layer_public_product_v1.py:1228-1249`);
2. G117 renders G105 directly for every pair
   (`g111_parsed_g105_stage_selector_v1.py:651-659`);
3. G117's G110 helper reparses the product, reopens the provider, and rerenders
   the pair (`taskspace_g110_generic_two_layer_public_product_v1.py:1252-1265`).

Across raw and Rice G105 wires this is roughly six n600 semantic passes before
the scorer work. Parse each packet once, render one population once, prove the
rank-zero equation from that parsed state, and reuse the population and hashes
for both outer ZIP methods. If raw and Rice parsed states/population hashes are
equal, reuse the scorer result only after that equality proof.

This is not merely speed polish: it keeps the full-n600 production screen well
inside the decoder/eval budget and reduces the number of independently
implemented receiver surfaces that can suppress signal.

## P1-2 — split immutable measurements from pointer-conditioned decisions

G117 includes the pointer identity in the batch-progress key
(`g111_parsed_g105_stage_selector_v1.py:1076-1098`). A pointer refresh therefore
invalidates expensive pixel/scorer work even when the semantic packet, labels,
scorer, V10 realization, and public runtime are unchanged.

Use two identities:

```text
measurement_key =
    H(packet, parsed state, public runtime tree, labels, scorer, V10, batch geometry)

decision_key =
    H(measurement_key, dynamic pointer snapshot, conditional policy)
```

The measurement store preserves exact disagreement counts and population
hashes without the pointer. The decision receipt adds the current target,
classifies retain/prune/defer, and post-verifies pointer stability. A changed
pointer requires a new decision receipt, not a redundant n600 scorer replay.

## P1-3 — use G115 as a measured regret/controller surface

The exact public-wire NumPy compiler and MLX straight-through surface already
exist in
`src/tac/witness_dsl/g105_public_wire_quantization_surface_v1.py:309-438`.
The G111 spec correctly states that semantic training acts on the pre-G105
floating state and that post-hoc parsed-wire selection closes the deployment
boundary; the trainer does not currently import G115.

For every retained physical stage, preserve both:

```text
d_seg_float_through_exact_R
d_seg_parsed_G105_through_exact_public_R
wire_regret = d_seg_parsed_G105 - d_seg_float
```

No guessed regret threshold should control terminal wire-QAT. A terminal,
resumable G105 wire-QAT stage is required before killing a stage when floating
state satisfies `100*d_seg_float < T` but parsed wire violates
`100*d_seg_wire >= T`. Otherwise wire-QAT may enter as another measured retained
stage and must win on exact downstream score value, not proxy loss.

## Verification

After G117 landed at `dbca26e2c7`, this review reran:

```text
uv run pytest -q -p no:cacheprovider \
  src/tac/witness_dsl/tests/test_g111_parsed_g105_stage_selector_v1.py
```

Result: `6 passed in 43.10s`.

A concurrent G117 producer report also recorded `41 passed in 93.92s` for its
focused G105/G110/G115/G117 matrix. That report is supporting context, not this
review's independently reproduced result.

During G117 work in progress, an earlier six-file aggregate produced 62 passes
and two failures caused by the then-changing helper name and newly mandatory
test-only fence. Those failures were corrected before the committed G117 suite
above passed. This review then reran the complete post-commit aggregate:

```text
uv run pytest -q -p no:cacheprovider \
  src/tac/witness_dsl/tests/test_g111_parsed_g105_stage_selector_v1.py \
  tools/tests/test_taskspace_g110_generic_two_layer_public_product_v1.py \
  src/tac/tests/test_v9_training_target_checkpoint_integration.py \
  src/tac/witness_control/tests/test_fresh_producer_lineage_v1.py \
  src/tac/witness_control/tests/test_taskspace_g112_exact_checkpoint_partition_v1.py \
  src/tac/witness_dsl/tests/test_v10_factor2_selected_preimage_v1.py
```

Result: `64 passed in 96.91s`. This is the load-bearing local
cross-component verification. It proves the reviewed interfaces and
fail-closed tests, not n600 scoring, public receiver closure, or a candidate.

A bounded Torch-CPU adversarial check of V10's sparse factor-2 realization
against `torch.nn.functional.interpolate(..., size=(384,512),
mode="bilinear")` used structured, seeded-random, and all-255 scorer arrays.
All cases were byte-identical (`array_equal=true`, maximum absolute difference
zero). This is structural support only, not evaluator or hardware authority.

## Triality and next score-facing action

DSL:

```text
PhysicalG112Stage
  -> ProductionParsedG105StageObservation
  -> RetainedSemanticStage[distortion-not-obstructed]
  -> ConditionalY0GivenY1Refit
  -> ReceiverClosedG110Archive
```

DAG:

```text
G46 -> G109 -> G111 stage deploy+resume -> G112 partition
    -> G120 production authority wrapper -> exact G105/G110 public-wire screen
    -> exact-obstruction retention set -> post-G105 conditional solve
    -> same-object archive race -> public double decode -> upstream/evaluate.py
```

Equation:

```text
V(Y1_i) =
    100*d_seg(Y1_i)
    + min_{Y0 | Y1_i, wire}
        [sqrt(10*d_pose(Y0,Y1_i)) + 25*archive_bytes(Y0,Y1_i,wire)/37_545_489]
```

The semantic screen does not estimate `V` by an arbitrary pose reserve. It
removes only stages whose first term already reaches the target, preserves the
rest, and measures the inner conditional value function exactly.

The next build should therefore be the production wrapper plus cross-stage
retention registry, followed immediately by a governed clean dry-run and the
first real retained G111 stage screen. If no stage crosses the strict
distortion obstruction, the producer—not the pose codec—is the measured
blocker. If at least one crosses, conditional refit and final archive eval
become the direct frontier action.

## Pointer delta

`NONE`. The canonical effective frontier observed by this review was the
external official display row `0.172`; this artifact did not create or evaluate
an archive. The mission remains unsatisfied until exact candidate bytes score
strictly below the dynamically refreshed competitive target.

## STORES CONSULTED

- `AGENTS.md`
- `CLAUDE.md`
- `.omx/state/canonical_frontier_pointer.json`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g111_batch16_v9_semantic_base_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g120_parsed_stage_production_authority_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g121_resumable_stage_harvest_controller_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/g110_generic_two_layer_public_product_receipt_20260727.json`
- `src/tac/witness_dsl/g111_parsed_g105_stage_selector_v1.py`
- `src/tac/witness_control/taskspace_g112_exact_checkpoint_partition_v1.py`
- `src/tac/witness_control/taskspace_v9_training_target_capsule_v1.py`
- `src/tac/witness_dsl/taskspace_g110_generic_two_layer_public_product_v1.py`
- `submissions/robust_current/g110_two_layer_receiver/inflate.py`
- `src/tac/witness_dsl/dynamic_frontier_target.py`
- `src/tac/witness_dsl/g105_public_wire_quantization_surface_v1.py`
- `src/tac/witness_dsl/v10_factor2_selected_preimage_v1.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`
