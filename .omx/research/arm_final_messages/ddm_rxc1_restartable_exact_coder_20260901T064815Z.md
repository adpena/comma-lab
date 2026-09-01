Verdict: **BLOCKED — Stage 2 is still executing**, so GATE-1-PASSED/FAILED cannot yet be adjudicated.

Measured so far:

- Baseline: exact 113,411-byte AFR1 stream reproduced with 0 differing bytes; 716.436 s.
- Null replay: 5 checkpoint starts across strides 200/300; 567,055 bytes compared; 0 differing bytes.
- Screen: 5/32 rows complete; exact full deltas `{2, 2, 5, 4, 4}` bytes. Row 5 is running from its retained frame-300 checkpoint.
- Active retained store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`
- Implementation: [RXC1 runner](/Users/adpena/Projects/pact/experiments/ddm_rxc1_restartable_exact_coder.py)
- Partial evidence memo: [RXC1 memo](/Users/adpena/Projects/pact/.omx/research/ddm_rxc1_restartable_exact_coder_20260901.md)

The runner remains active and resumable. The memo and source are intentionally uncommitted until `SCREEN.json`, the final manifest, tests, and correlation verdict exist.

Own-vehicle frontier: **UNMOVED — AFR1 S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]**. RXC1 is scorer-free and makes no score claim.

## NEXT_IF_RESUMED

- `FIRED/RUNNING` — owner: `ddm_rxc1`; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger: existing Stage-2 process completion or interruption; action: harvest/resume the frozen n=32 screen, generate `SCREEN.json` and `MANIFEST.json`, finalize the memo, run verification, and invoke the mandated commit serializer.

## LIVE-HYPOTHESES

- Exact incremental replay will equal full re-encoding for nonzero checkpoint starts because the null gate proved complete HPAC/corrector and RC64 state restoration.
- Pearson and Spearman will clear 0.9 if that stream identity persists; the first five full deltas already provide nonconstant variation.
- Later checkpoint starts should reduce cost approximately in proportion to suffix length, consistent with the measured 475.067/355.997/237.690-second null replays.

## DEAD-ENDS

- Cheap differentiable/static rate surrogates remain closed by CM1; none was substituted here.
- A second coder implementation was rejected because JG2 already provides the physical AFR1 mirror.
- Partial corrector or range-only checkpoints were rejected because they cannot reproduce the causal HPAC stream.
- Duplicate frame-0 incremental executions were removed: checkpoint 0 is the already-executed full exact run, so recomputing it would add no evidence. [no-triality] [p0-ledger-ok]