Implemented a real partial substrate, but the charter is not complete. Full mode fails closed before training because the named solve and receiver stages cannot currently satisfy the required per-clip/device contracts.

Key artifacts:

- [Research handoff](/Users/adpena/Projects/pact/.omx/research/ddm_fpc1_full_pipeline_compress_20260903.md:1)
- [Pipeline implementation](/Users/adpena/Projects/pact/src/tac/semantic_pipeline/pipeline.py:1)
- [Stage contracts](/Users/adpena/Projects/pact/src/tac/semantic_pipeline/contracts.py:1)
- [CLI](/Users/adpena/Projects/pact/experiments/semantic_joint_ctxmix_pipeline.py:1)
- [Tests](/Users/adpena/Projects/pact/src/tac/tests/test_semantic_pipeline.py:1)

Measured:

- Real `0.mkv` probe: 1,200 frames, 1164×874, 600 non-overlapping pairs.
- Retained replay revalidated all six stage payloads and produced the exact 180,002-byte AFR1 archive, SHA `cbb8d928…d405bf25`.
- This was payload/hash revalidation, not a fresh 4,140.9-second mechanism replay.
- `pytest`: 5 passed in 0.83s.
- Ruff: clean.
- No scorer, n600 training, Modal, MPS, or CUDA run occurred.

The full-mode blocker receipt is `/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/full/RESULT.json`. It identifies six concrete blockers, including QS5’s hardcoded historical object, absent solve-stage device flags, an AFR1-only/CUDA-only shipped receiver, an unreachable F26 prefix branch, incomplete trainer EMA/device contracts, and the newly recalled DALI/AV target-lineage confound.

The intended commit could not land because the managed sandbox denied Git object writes. The serializer preserved fallback commit `2296c6bad8d741303e64b065ea8fee09d3117160` in the verified [bundle](/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/receipts/commit_serializer_fallbacks/20260903T171959.951976Z-19152/intended-commit.bundle). The six files remain untracked and the shared index is empty.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER** — owner: `fpc1 successor`; consumer store: `src/tac/semantic_pipeline/` and `/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress`; action: parameterize FCD1/JG5/QS5/UP2 and build the fresh-archive CPU prefix receiver; fire trigger: real n=2 receiver identity passes with derived EMA, exact roundtrip/YUV6, and retained payloads.
- **QUEUED-WITH-FIRE-ORDER** — owner: `MAIN`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress`; action: land fallback commit `2296c6b` and later fire the governed n600 ticket; fire trigger: ports pass, a numeric memory-preflight receipt exists, and MAIN owns the Metal lane.
- **QUEUED-WITH-FIRE-ORDER** — owner: `MAIN`; consumer store: the future governed T4 evaluation store; action: evaluate the fresh receiver-closed archive; fire trigger: n600 full mode completes and MAIN claims the unique scorer lane.

## LIVE-HYPOTHESES

- A faithful pipeline remains feasible by extracting parameterized kernels from the named scripts; much of their mechanism is reusable even though their orchestration is instance-pinned.
- A CPU prefix receiver is plausible because F26 already contains pair-limit, checkpoint, and parallel-render machinery; its contradictory token-decoder gates need resolution.
- A fresh PR130-derived archive can be valid, but it must explicitly choose DALI or AV targets and receive a fresh score.
- The lifted PyTorch renderer could support CPU/MPS/CUDA once device propagation, derived EMA, exact roundtrip, and differentiable YUV6 are integrated.

## DEAD-ENDS

- Simple CLI chaining is closed: the solve scripts lack common archive/per-clip/device inputs.
- The shipped top-level receiver cannot check a fresh CPU archive: it pins AFR1 and requires CUDA.
- `F26_ADVISORY_PAIR_LIMIT=2` is closed: prefix mode requires native-hpac after native-hpac has already been refused.
- Retained-payload adoption cannot be called a fresh replay.
- A fresh single-DALI run cannot be called a bit-identical reconstruction of the mixed historical PR130 target lineage.

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600] — unchanged; this arm produced no new exact score.**