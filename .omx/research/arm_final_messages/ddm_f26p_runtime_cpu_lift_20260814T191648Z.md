Implemented and landed the F26 CPU lift in commit `a5e1f60270`.

Key results:

- Full n600 decode completed in **646.416 s** on `[M5-CPU 4-thread LOWER BOUND on contest wall]`: `LIKELY-IN-BUDGET`.
- Retained raw: 3,662,409,600 B, SHA `e5539653…`.
- Retained token checkpoint: 117,964,800 B, SHA `9ba2e52b…`.
- CPU raw differs from T4 SHA `a41ca69d…`; frame count and maximum delta remain unquantified because the T4 bytes are only on the Modal volume.
- No evaluator ran and no score was claimed. The frontier remains **MC36 Variant C, S=0.1619344578804448 @ 186,269 B `[contest-CUDA T4,n600]`**.
- The sealed CUDA runtime was not modified.

Artifacts:

- [Handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_f26p_runtime_cpu_lift_20260814.md)
- [CPU runner](/Users/adpena/Projects/pact/experiments/ddm_f26p_f26_cpu_lift.py)
- [Lifted decoder](/Users/adpena/Projects/pact/experiments/ddm_f26p_f26_inflate_cpu.py)
- [Result receipt](/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/receipts/result.json)
- [Runtime analysis](/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/receipts/runtime_analysis.json)

Verification passed: Ruff, Python compilation, 34 payload-retention tests, payload-discard audit, two review-tracker passes per Python file, full retained decode, and post-commit SHA verification.

## NEXT_IF_RESUMED

- **QUEUED** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/receipts/`; fire trigger: download T4 raw SHA `a41ca69d…` from `comma-ddm-js1b-argmax-retained`, then run the documented `finalize --t4-raw` comparison.
- **QUEUED WITH FIRE ORDER 2** — owner: MAIN; consumer store: `experiments/results/ddm_f26p_mc36_contest_cpu_20260814/`; fire trigger: reconcile the earlier MC36 CPU lane terminally, claim the new lifted-runtime lane, confirm no live Modal job, then use the exact command in the memo.
- **QUEUED CONDITIONAL** — owner: successor CPU-runtime arm; consumer store: `receipts/runtime_analysis.json`; fire trigger: contest CPU exceeds 1,500 s or times out; prototype gathered-one-hot conv-a with full logit, CDF, token, and raw parity.

## LIVE-HYPOTHESES

- The lifted runtime may fit a favorable contest CPU because the local full wall is 646.4 s, though prior lineage measurements show substantial Modal host variance.
- MC36’s CPU score may differ materially from CUDA because the raw bytes already differ; the related lc2 CPU regression makes a worse result plausible, but does not prove it.
- Gathered-one-hot conv-a could reduce its arithmetic count by up to 7× while preserving tokens, provided changed summation order survives exact parity.

## DEAD-ENDS

- Existing-stream four-worker decode: F26 has one RC64 stream with causal group and frame dependencies.
- Another native RC64 rewrite: RC64 is already native C and previously measured below 0.5% of token wall.
- MLX or Metal in contest-CPU decode: those assets serve local training and screening only.
- CPU/T4 byte identity: aggregate hashes conclusively differ.
- Any score or frontier claim from this arm: no exact evaluator ran.