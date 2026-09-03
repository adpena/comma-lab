# DDM FPC3 — chunked n600 trainer ready, population fire queued

## Outcome

The from-raw-video pipeline now has a real all-600-pair training consumer. It materializes and retains
the AV-decoded RGB pair cache and frozen-SegNet target cache, trains in an explicit seeded
`stratified_blocks` or `seeded_random` permutation, and writes an atomic checkpoint after every chunk.
Each checkpoint carries the live model needed to continue, the EMA shadow used to ship, optimizer
state, Python/NumPy/Torch/device RNG state, the next-chunk cursor, every trajectory-defining config
value, and content facts for the trainer, archive builder, roundtrip, scorers, scorer weights, and
receiver. The final scorer forward is independently resumable and bounded by `--verdict-batch <= 120`.

The bounded FPC2 path remains the explicit n<=8 prefix plumbing smoke. The population path requires
exactly 600 pairs, CUDA, 6,000 updates, a concrete unique live scorer claim, a PASS fire-time
memory/storage admission, and its governed ticket. No n600 run, Metal/MPS run, Modal call, or launch
was made by this arm.

This implementation follows the stage-boundary and custody rules in
`docs/operating_manual_craft_handoff.md`: payloads are retained under the named Vertigo store,
interruption resumes from an explicit durable boundary, and promotion remains downstream of exact
archive/evaluator/hardware custody.

## Measured resume identity

Receipt:
`/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/resume_smoke/919db3b19bf4f32b/RESUME_IDENTITY_RESULT.json`
(sha256 `03bdec8e7c5b5b7bf23ae4214989ddca51bab8279b750b05ef7722a789ad2b72`).

- Axis: `[macOS-CPU exact-scorer n6 mechanism smoke; not a verdict]`; contiguous n=6 prefix only for
  mechanism proof; `score_claim=false`.
- Three B=2 chunks were `[2,3]`, `[0,5]`, `[1,4]`, a seeded-random permutation rather than video order.
- Interrupt after chunk 1 plus resume matched the uninterrupted three-chunk endpoint on all four
  registered objects: live state, EMA state, archive bytes, and loss history.
- Both endpoints produced archive sha256
  `61fa8bbc9d21344f3c130ba40d826903f184c138cacea503b49e63ffd10f7f52`, 181,460 B.
- The retained bounded scorer read was identical at `d_seg=0.0002772013346354167` and
  `d_pose=0.000009364848766482384`. These are prefix mechanism-smoke values, not population evidence,
  not an exact contest score, and not a frontier row.
- The receipt reports 88.41143287532032 s for six executed updates across the two trajectories.
  Camera, SegNet argmax, PoseNet pose6, checkpoints, caches, model states, and both archives remain
  under the receipt root.

The strict `check_ema_executable_law_matches_sealed_law` live count is 0. The executed and sealed law
is `ema_decay_run_geometry_v1`, constant decay derived from terminal seed fraction 0.1; the
constructor consumes the typed `warmup=false` field and contains no literal `warmup=` choice.

## Memory and storage admission

Memory receipt:
`/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/MEMORY_PREFLIGHT.json`
(sha256 `e6ae71159ba9c9e99de84bdd98f87d10d64355a89294a1926c902e773a6fc94e`).

- The measured basis is QBR1's real-scorer B=16 resume proof:
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/resume_smoke/RESUME_SMOKE_RESULT.json`,
  45,475,168,256 B = 45.475168256 GB decimal = 42.35205078125 GiB peak RSS, with 4/4 endpoint
  equality.
- Real FPC3 config: 600 pairs, training B=16, verdict B=32. Projected process peak is
  45,475,168,256 B under the measured-B16 floor / linear-above-B16 rule.
- Host RAM was 137,438,953,472 B; the binding 0.70 ceiling was 96,207,267,430 B. Current use was
  71,116,603,392 B, so current plus projected peak was 116,591,771,648 B. Status: `REFUSE`.
- This refusal is expected while QBR1 owns the Metal/scorer resources. The population command repeats
  the same system-aware admission at fire time and cannot proceed on `REFUSE`.

Storage receipt:
`/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/STORAGE_PROJECTION.json`
(sha256 `17436cad9e0a015b41dabe475013d80691c58fee5a85e8edd19021d8cef53cba`).

- Retained subtotal: 15,028,122,000 B; with 25% margin: 18,785,152,500 B.
- The projection prices all 6,000 per-chunk checkpoints, the 3,662,409,600 B AV RGB cache, the
  117,964,800 B semantic target cache, final camera/argmax/pose payloads, and the 3,662,409,600 B
  receiver raw output.
- Vertigo free space was 193,351,880,704 B. AP free space was 31,009,013,760 B, preserving the
  required 8,589,934,592 B reserve. Status: `PASS`.

## Governed launch ticket

Ticket:
`/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/governed_n600_launch_ticket.json`
(sha256 `4d93a62d417f353a83f08616836cc8b4d5c8d892e5965f4c5e001143da154717`). The exact argv passed the
launcher's `--dry-run` parser; it was not launched.

```text
.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/launch/n600_cuda \
  --purpose "FPC3 crash-resumable chunked n600 scorer-aware CUDA population run" \
  --authority "candidate contest-CUDA; exact hardware and evaluator receipts decide promotion" \
  --derive-resource-budgets \
  --measured-peak-rss-gib 42.352050781 \
  --measured-thread-need 4 \
  --walltime-cap-s 210246 \
  --done-receipt ddm_fpc3_n600_cuda \
  -- \
  .venv/bin/python experiments/semantic_joint_ctxmix_pipeline.py \
  --mode full --device cuda \
  --video /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --store /Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer \
  --seed 20260903 --pairs 600 --updates 6000 \
  --chunk-pairs 16 --selection-mode stratified_blocks --stratified-blocks 10 \
  --verdict-batch 32 \
  --scorer-claim-id MAIN_MUST_INSERT_UNIQUE_SCORER_CLAIM_ID \
  --resume-from latest
```

Timing is measured-basis projection, not a CUDA measurement: QBR1's 112.13081141607836 s covered four
executed B=16 CPU scorer updates, or 28.03270285401959 s/update. At 6,000 updates that projects
168,196.21712411754 s = 46.72117142336598 h; the launcher cap is 1.25x = 210,246 s.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_semantic_pipeline.py -q`: 12 passed.
- Ruff: clean on the trainer, pipeline, CLI, and acceptance test.
- Python bytecode compilation: clean on the same four files.
- Two genuine review-tracker passes recorded on every changed Python file.
- `git diff --check`: clean.
- Upstream and `submissions/semantic_joint_ctxmix/` were not edited.

## RECALL EVIDENCE

Searched the full `.omx/research` and `.omx/state` Markdown corpus by content for
`ddm_fpc3|chunked n600|QBR1.*45.5|45.5 GiB|resume_smoke`, then searched the canonical research index,
the main `sub015_DAG` FEED, and task-ledger surfaces for
`semantic_joint_ctxmix|fpc2|fpc3|chunked.*n600|ema_decay_run_geometry|verdict.batch|prefix bias|1390`.
Also ran `.venv/bin/python tools/list_canonical_equations.py --json` and inspected the live lane board.

Beyond the charter seeds, the DAG's FEED-oom records the measured #205 distinction: the OOM was the
single population-wide verdict transient, while `--verdict-batch 32` collapses it to the chunked floor.
That made verdict chunking a separately checkpointed consumer rather than merely a training batch
option. The older j3 fire chain records that a built full-run path is still not fireable without a
fresh measured timing seal and live preflight; that changed this ticket from a descriptive estimate to
a dry-run-validated launcher command with a fire-time admission receipt. The equation registry resolved
the EMA value through `ema_decay_run_geometry_v1`; no alternate FPC3 owner or direct task-#1390
implementation was found in the bounded index/task-ledger scope.

## NEXT_IF_RESUMED

- `QUEUED-WITH-FIRE-ORDER` — owner: MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/full`; fire trigger: the QBR1 burn writes
  terminal Metal and scorer rows, MAIN appends one unique active scorer claim, reruns the FPC3 launch
  preparation and requires memory plus storage `PASS`, replaces the ticket's scorer-claim placeholder
  with that exact id, then executes the retained argv.

## LIVE-HYPOTHESES

- The CPU-proven cursor/RNG/EMA contract will reproduce a CUDA interrupted population endpoint
  bit-for-bit because every observed hidden trajectory input is now sealed and restored. This remains
  untested on CUDA and at n600.
- The system-aware memory gate will change from `REFUSE` to `PASS` after QBR1 releases enough unified
  memory because the standalone 42.35205078125 GiB projected trainer peak is below the 89.6 GiB safety
  ceiling; current co-resident use, not the standalone projection, caused this refusal.
- The complete 600-pair stratified training run remains worth firing because the only missing evidence
  is the population trajectory and exact receiver/evaluator result; the n=6 proof established
  apparatus identity, not quality or frontier movement.

## DEAD-ENDS

- Firing the ticket now is closed at INSTANCE scope: QBR1 still owns the live scorer/Metal rows and the
  current system-aware receipt is `REFUSE`.
- Treating the n=6 receipt as a score is closed: it is a contiguous-prefix mechanism smoke with no
  upstream population evaluation and `score_claim=false`.
- Video-order prefix training is closed: population chunks require an explicit seeded-random or
  stratified selector, and the acceptance control rejects prefix order.
- An unchunked 600-pair scorer forward is closed: the public config and trainer cap verdict batches at
  120, default 32, consuming the measured #205 OOM law.
- Loop-end-only checkpoints and live-weights-as-EMA are closed: every optimizer update writes the live
  continuation state and distinct EMA shadow before advancing the cursor.
- A literal EMA warmup switch is closed: the typed `ema_decay_run_geometry_v1` seal drives execution and
  the strict confound gate remains at live count 0.

OWN-VEHICLE FRONTIER: UNMOVED — FPC3 has no n600 score; canonical effective frontier remains
`S=0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]` (AFR1).
