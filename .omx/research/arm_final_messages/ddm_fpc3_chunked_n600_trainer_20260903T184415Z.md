Implemented and committed FPC3 as `7aada29099863e2a54306354b626c70d45e3a675`.

The population path now provides:

- Seeded random/stratified all-600-pair chunk scheduling.
- Atomic per-chunk checkpoints with live state, EMA shadow, optimizer, RNG, cursor, config, and implementation hashes.
- Resumable AV/semantic-target materialization and chunked scorer verdicts.
- Typed `ema_decay_run_geometry_v1`; strict EMA-law gate remains at 0 findings.
- Fire-time memory, storage, and unique scorer-lane enforcement.
- Governed detached CUDA ticket with `--resume-from latest`.

The durable handoff and exact command are in [ddm_fpc3_chunked_n600_trainer_20260903.md](/Users/adpena/Projects/pact/.omx/research/ddm_fpc3_chunked_n600_trainer_20260903.md:1). Core implementation: [train.py](/Users/adpena/Projects/pact/src/tac/semantic_pipeline/stages/train.py:1084) and [pipeline.py](/Users/adpena/Projects/pact/src/tac/semantic_pipeline/pipeline.py:94).

Measured:

- CPU n=6, three-chunk resume proof: live state, EMA state, archive, and loss history all identical.
- Identical archive: 181,460 B, SHA-256 `61fa8bbc9d21344f3c130ba40d826903f184c138cacea503b49e63ffd10f7f52`.
- Acceptance: 12 tests passed; ruff, compilation, two review passes, and launcher dry-run passed.
- Memory preflight: `REFUSE` while QBR1 is resident—116,591,771,648 B projected system use versus the 96,207,267,430 B ceiling.
- Storage: `PASS`—18,785,152,500 B projected with margin; AP’s 8 GiB reserve remains intact.
- No n600 training, Metal/MPS, Modal, or launch occurred. FPC3 has not moved the frontier.

## NEXT_IF_RESUMED

- `QUEUED-WITH-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/full`; fire trigger: QBR1 writes terminal Metal/scorer rows, MAIN registers one unique active scorer claim, reruns preparation with memory and storage both `PASS`, substitutes that claim ID into the retained ticket, and executes its argv.

## LIVE-HYPOTHESES

- CUDA n600 interruption will reproduce bit-identically because every observed trajectory input is sealed and restored; this remains untested at population scale.
- Memory admission should pass after QBR1 releases sufficient unified memory because the standalone projected peak, 42.352 GiB, is below the 89.6 GiB ceiling.
- The population run remains worth firing because apparatus identity is proven; training quality and exact receiver/evaluator outcome are the remaining unknowns.

## DEAD-ENDS

- Immediate launch is closed: QBR1 owns the resources and current memory admission refuses.
- The n=6 values cannot be treated as a score: they are prefix mechanism evidence with `score_claim=false`.
- Prefix population training, unchunked n600 verdicts, loop-end-only checkpoints, live-weights-as-EMA, and literal EMA warmup selection are mechanically refused.
- FPC3’s own-vehicle frontier remains unmoved; the canonical effective frontier is `S=0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]` (AFR1).