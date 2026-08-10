# ddm_sd2 PR130 retained seg-decomposition runner

## Outcome

The paired full-population n600 runner is built and locally validated. It retains the exact
public-receiver camera outputs, target camera chunks, full SegNet logits, SegNet argmax, and full
PoseNet outputs before computing any scalar or decomposition. The 5x5 directed matrix, per-frame
mass, directed and symmetric edges, and boundary/interior split are recomputed only after the
argmax chunks are reopened and their recorded SHA-256 values are verified.

No scorer, PoseNet, SegNet, real receiver, or contest evaluator ran in this arm. No d_seg, d_pose,
edge count, frame count, boundary rate, S value, or pointer movement was measured. `score_claim`
is false.

The exact charter storage target is mounted with `1,070,843,559,936` free bytes, and the fresh-fire
requirement is `27,862,199,312` bytes including a `3,662,409,600`-byte failed-decode contingency and
a `5,000,000,000`-byte reserve. Capacity fits. This codex sandbox nevertheless receives
`PermissionError: [Errno 1] Operation not permitted` when it attempts to create the run directory.
The runner now fails closed at that point and will not fall back to Vertigo or a symlink.

## Retention proof

- Runner: `experiments/ddm_sd2_pr130_seg_decomposition_runner.py`, 78,536 bytes, SHA-256
  `29e88f44232322a27d1376b9401a9ed1a89eec6c1ed1f1fdf1bc600d66d6a576`.
- Tests: `experiments/tests/test_ddm_sd2_pr130_seg_decomposition_runner.py`, 8,721 bytes, SHA-256
  `03000dfbda0ab8b305bf38178ee7562bf0ba8efb1ad790e34849c4132c41a57f`.
- `pytest`: 12 passed. `ruff check`, Python compile, and `git diff --check`: PASS.
- `tac.payload_retention_gate`: 8 files scanned, 0 findings. Scope was the runner, its tests, and
  the six pinned Python receiver modules materialized from commit
  `58f62cd22ff07562c0534c999d705fb9edfe5279`.
- The writer test executes real atomic byte, NumPy memmap, byte-range hash, and JSON-receipt probes.
  Tests also cover interrupted full-chunk finalization, contiguous-prefix resume refusal, retained
  NPY rehash refusal, stale decode/archive binding refusal, the n600-only measurement rule, and
  APDataStore-only bulk routing.
- Machine-readable proof: `.omx/research/ddm_sd2_20260810/SD2_RETENTION_PROOF.json`.

The projected final retained footprint is `19,199,789,712` bytes:

| Retained component | Bytes |
|---|---:|
| two exact receiver raw outputs | 7,324,819,200 |
| target decoded camera chunks | 3,662,409,600 |
| target/base/candidate SegNet float32 logits | 7,077,888,000 |
| target/base/candidate SegNet uint8 argmax | 353,894,400 |
| target/base/candidate PoseNet float32 outputs | 86,400 |
| token checkpoints, dependencies, receipts, archive/repeat copies | 780,692,112 |

The earlier scorer-free development preflight under the writable Vertigo sandbox produced only
1,283,713 logical bytes across 19 `ddm_sd2` files. They remain retained because the charter forbids
deletion. They are not the fire target and are not presented as authority evidence.

## Exact MAIN fire command

```bash
/Users/adpena/Projects/pact/.venv/bin/python /Users/adpena/Projects/pact/experiments/ddm_sd2_pr130_seg_decomposition_runner.py --out-dir /Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600 --resume-from /Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600/progress.json --queue /Users/adpena/Projects/pact/.omx/research/ddm_sg2_20260810/SG2_SCORER_QUEUE.json --base-archive /Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip --candidate-archive /Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip --challenge-root /Users/adpena/Projects/pact/upstream --video-names-file /Users/adpena/Projects/pact/upstream/public_test_video_names.txt --uncompressed-dir /Users/adpena/Projects/pact/upstream/videos --device mps --decode-device cpu --batch-size 4 --chunk-pairs 60 --pair-count 600 --seed 20260810 --cpu-threads 6 --num-threads 2 --prefetch-queue-depth 4 --minimum-free-bytes 5000000000
```

Fire order: `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN scorer owner`; consumer store
`/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600`; trigger: the committed runner is
present, APDataStore is writable to MAIN, and the sole full-n600 scorer slot is free. The prior
same-family single public-receiver decode took 1,010.808 seconds on the
`[macOS-CPU advisory; upstream AV GT; immutable evaluate.py; n600]` axis; the honest projected band
for two serial CPU decodes plus the unmeasured three-way retained MPS pass is approximately 45-90
minutes.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus by content for `PR130`, `argmax`, `retention`, `SegNet`,
`directed edge`, `boundary`, `interior`, `q3`, and `q4`; queried the canonical-equations registry
with `.venv/bin/python tools/list_canonical_equations.py --json`; searched
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, the task ledger, v7.5/v8 specifications,
the live hot state, and the actual upstream evaluator and scorer implementations.

Beyond the charter seeds, the search found:

- The live `src/tac/pr130_runtime/dv1_cpu_runtime/{inflate.py,receiver.py}` files contain unrelated
  uncommitted temporal-reversion work. That changed the runner to materialize every receiver file
  directly from Git commit `58f62cd22ff07562c0534c999d705fb9edfe5279`, checking a fixed per-file
  SHA-256, rather than consuming the dirty worktree.
- `.omx/research/ddm_cx2_20260809/CX2_FINAL_RECEIPT.json` records a 1,010.808-second full n600 decode
  for this receiver family. That replaced an ungrounded runtime guess with the 45-90 minute fire
  band while leaving the new three-way scorer portion explicitly unmeasured.
- `experiments/ddm_pc2_edge_decomp.py` is a different-vehicle cached-flip reducer. It confirmed the
  importance of directed edge accounting, but its cached taxonomy is the wrong object for PR130.
  The new runner therefore builds its matrices from the retained PR130 target/prediction argmax and
  defines boundary directly on the retained target by a two-sided four-neighbor transition.
- The equation/index/DAG/ledger searches did not find another retained PR130 candidate argmax in
  the searched scopes. This preserves SG2's bounded absence and keeps the scorer fire queued rather
  than pretending the decomposition can be recovered from rounded scalar output.

## Boundaries and frontier

The future MPS row is diagnostic, never contest authority. The PR130
`S=0.17214129749189645 @ 191,052 B [contest-CUDA, DALI GT, n600]` row remains a borrowed bar. This
arm did not move any exact pointer. The live own-vehicle line therefore remains
`S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`.
