# ddm_rr16 ROUND16_FINDINGS

## Verdict

NOT-CLEAN. rr16 resets the clean-advance counter to 0/3.

The mx1t tail-average values are reproducible under the ARM-CAP cache binding, but
the committed mx1t findings/receipt rows do not carry the load-bearing
input/target cache identity, cache hashes, or replay argv needed to reproduce the
facet series without hidden context. That is a policy-blocking provenance defect:
do not consume the mx1t tail-average selection policy until a cache-bound repair
receipt is committed.

This is not an exact score, not an n600 result, and not a frontier move. Axis for
all reruns below is `[macOS-CPU advisory torch upstream SegNet]` with
`score_claim=false`.

## F1 - CRITICAL - Cache Binding Missing From Committed Receipts

The committed facet rows are not self-contained enough for a load-bearing
selection policy. Replaying the copied step-3250 checkpoint with the parser
defaults uses `tq1c_seg_cache.pt` as input and `gt_seg_cache.pt` as target, and
measures:

- default-cache repeat: `d_seg = 0.004567305246988933`, expected
  `0.0010732014973958333`, absolute difference
  `0.0034941037495930995`
- blocker:
  `.omx/research/ddm_rr16_20260807/runtime_checks/repeat_final_a/MX1T_ANCHOR_BLOCKER.json`

The original ARM-CAP mx1t run instead used GT cache for both input and target.
That binding is visible in the untracked `mx1t_facets_result.json` and launch
evidence, but it is not present in each committed JSONL facet row. With the
correct binding, two independent repeats of the copied step-3250 checkpoint
reproduce bit-identically:

| run | input cache | target cache | measured d_seg | mismatch pixels | abs diff |
| --- | --- | --- | ---: | ---: | ---: |
| mx1t committed row | gt_seg_cache.pt | gt_seg_cache.pt | 0.0010732014973958333 | 6752 | 0 |
| rr16 repeat A | gt_seg_cache.pt | gt_seg_cache.pt | 0.0010732014973958333 | 6752 | 0 |
| rr16 repeat B | gt_seg_cache.pt | gt_seg_cache.pt | 0.0010732014973958333 | 6752 | 0 |
| rr16 default-cache repeat | tq1c_seg_cache.pt | gt_seg_cache.pt | 0.004567305246988933 | not accepted | 0.0034941037495930995 |

Cache hashes verified in rr16:

- `gt_seg_cache.pt`:
  `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`
- `tq1c_seg_cache.pt`:
  `11fd89016ab33a4b221975dafac0a572d66c372db1accf62284a3e81acddcc54`

Required repair before policy consumption: commit a cache-bound receipt/addendum,
or patch the facet writer and rerun, so every load-bearing facet row records
input cache path, target cache path, SHA-256 for both caches, selected pair IDs,
source repo head, replay argv, and score-claim axis. The existing mx1t numbers
may be carried as measured ARM-CAP/GT-GT advisory values after that repair; until
then, the selection policy is HOLD.

## Axis Review

### 1. Tail-average implementation

No averaging-over-wrong-tensors defect was found for the copied mx1t corpus.
`_load_mlx_npz_checkpoint_for_torch` validates the NPZ metadata and expected
model state dict, rejects missing/unexpected parameter names, and shape-checks
tensors before loading them into the torch renderer. All 13 copied checkpoints
have matching NPZ key sets. `_load_average_torch_checkpoint` requires identical
config JSON and pair IDs across members, averages floating state-dict tensors,
and copies non-floating tensors from the latest member. The tail-average rows are
then evaluated through the same `_mx1t_evaluate_checkpoint_facets` path as normal
checkpoints.

That leaves F1 as a replay/provenance defect, not a tensor-averaging defect.

### 2. Statistical honesty of the K=8 win

The committed K=8 tail-average row is an n32 advisory win over the final
checkpoint:

- final step 3250: `d_seg = 0.0010732014973958333`, 6752 mismatch pixels
- K=8 tail average: `d_seg = 0.0010673205057779949`, 6715 mismatch pixels
- delta: `-0.000005880991617838397`, or 37 fewer mismatch pixels over
  6,291,456 evaluated pixels

The rr16 repeated step-3250 measurement had zero observed repeatability
difference under the correct cache binding. Therefore the K=8 delta is above the
instrument repeat floor observed by this review. It is still only n32 and may not
be stated as an n600, exact, public-wire, or population-tail result.

### 3. Facet metric definitions and rederived verdict

The facet metric definitions are coherent for this scope:

- per-class rows separate GT-side miss rates from prediction false positives
- margin histograms use fixed bins over mismatched pixels and correct boundary
  band pixels
- churn is normalized by the current mismatch-pixel denominator
- class order is the canonical comma10k order, not luma-sort

The first-to-final facet trend rederives from the receipts:

- near-miss fraction rises from `0.39800393164978076` to
  `0.4022511848341232`
- far-margin fraction falls from `0.06381370028731287` to
  `0.057760663507109004`
- low-churn median is `0.035615937141328075`
- first checkpoint beats final checkpoint on aggregate d_seg, so the only
  positive mx1t next-step signal is the tail-average row, not "more steps"

### 4. Checkpoint-copy custody

All 13 committed copy receipts were checked against files under
`.omx/research/ddm_mx1t_20260807/checkpoint_copies/`. File sizes and SHA-256
values matched the receipt rows, and the step-3250 copy hash was reproduced in
the rr16 repeat copies:

`543ad45a32ae6d4ac84cfd7fbb6845afb0c7bfc29525992dcf9744ec7540c382`

The source repo head recorded in facet rows and rr16 repeats is:

`2f94596bb0136d342254022a5c9584756eae0468`

### 5. Repair integrity of commit 0bc7c20966

`0bc7c20966` only adds/modifies mx1t-owned files:

- `.omx/research/ddm_mx1t_20260807/CHARTER.md`
- `.omx/research/ddm_mx1t_20260807/MX1T_FINDINGS.md`
- `.omx/research/ddm_mx1t_20260807/mx1t_checkpoint_copy_receipts.jsonl`
- `.omx/research/ddm_mx1t_20260807/mx1t_facets_receipts.jsonl`
- `experiments/ddm_mx1_pr130_semantic_renderer.py`
- `experiments/tests/test_ddm_mx1_memory_probe.py`

No active-lane ledger row was committed by that MAIN repair commit. The dirty
working tree still contains uncommitted mx1t result JSON files and unrelated
work; rr16 did not revert or fold those into this review.

### 6. Per-round assumption challenge

The challenged assumption was that a copied checkpoint plus committed facet row
was enough to replay mx1t. That was false: the parser default cache binding
changes the measurement by `0.0034941037495930995`. The correct ARM-CAP/GT-GT
binding makes the anchor reproduce exactly, which is why the repair route is
provenance-hardening rather than rejecting the measured mx1t row outright.

## Recall Evidence Beyond The Seed List

This review searched beyond the charter seed files and used the following
additional scope:

- `.omx/state/main_hot_state.md` and `.omx/state/canonical_frontier_pointer.json`
  for the live pointer and no-score-claim boundary
- `.omx/state/active_lane_dispatch_claims.md` for the dirty/uncommitted mx1t
  completed-lane row
- `.omx/research/arm_final_messages/ddm_mx1t_20260807T215947Z.md` for the arm's
  own report that active-lane rows and result JSONs were left uncommitted
- `.omx/research/ddm_mx1g_20260807/LAUNCH_TICKET.json` and related launch
  receipts for the ARM-CAP GT/GT cache binding
- canonical equation/review-tracker references for prior tail-average work, so
  mx1t is treated as a scoped simple tail-average measurement rather than a
  general plateau-law proof

## Boundary

No live run directory was mutated. No Metal or MPS authority was used. No
`upstream/evaluate.py`, archive replay, n600 scorer run, or public-wire score was
claimed. rr16 performed bounded CPU torch advisory replays of copied
checkpoints, receipt/custody checks, and commit-diff review only.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
