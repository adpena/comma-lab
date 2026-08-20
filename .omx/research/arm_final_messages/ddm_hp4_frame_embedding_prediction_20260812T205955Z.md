HP4 is complete and committed as `507da6e9fc`. The state-prediction hypothesis is closed for the tested formulation.

| Treatment | Complete archive | Delta |
|---|---:|---:|
| CP135 incumbent | 186,252 B | — |
| Order-0 Brotli q11 | **186,247 B** | **-5 B** |
| AR1 | 186,419 B | +167 B |
| Partition state | 186,511 B | +259 B |
| Pose/carrier state | 186,523 B | +271 B |
| Joint state | 186,607 B | +355 B |
| AR2 | 186,835 B | +583 B |

Every actual state-conditioned predictor enlarged the complete container. The only win was a no-prediction order-0 repack, so it does not support the charter hypothesis.

Verification:

- 36 complete-container candidates; all payloads and repeats retained.
- 396 artifact records independently verified; 485 files in the immutable manifest.
- Exact embedding reconstruction and unchanged semantic/carrier/residual/token streams passed.
- No scorer, renderer, `evaluate.py`, Modal, MPS, CUDA, or network operation ran.
- Contest pointer remains CP135 at `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; the derived five-byte rate improvement was not promoted.

Artifacts: [landed memo](/Users/adpena/Projects/pact/.omx/research/ddm_hp4_frame_embedding_prediction_20260812.md), [result](/Volumes/VertigoDataTier/pact/ddm_hp4/retained/FINAL_RESULT.json), [custody receipt](/Volumes/VertigoDataTier/pact/ddm_hp4/retained/FINAL_CUSTODY_RECEIPT.json), [immutable manifest](/Volumes/VertigoDataTier/pact/ddm_hp4/retained/FINAL_IMMUTABLE_MANIFEST.json).

## NEXT_IF_RESUMED

- `order0_brotli_q11_repack` — disposition: `FOLDED`; owner: `MAIN / next CP135 probability-object composer`; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/`; fire trigger: construction of the next score-bearing probability-object archive. Recount against that parent and retain only if receiver parse-back remains exact and the complete-container delta remains negative.

## LIVE-HYPOTHESES

- The -5 B order-0 repack may survive composition because it won at the real ZIP boundary, though its margin is small enough for container interactions to erase.
- Jointly retraining the probability object to produce a prediction-friendly embedding remains plausible because it changes the representation itself; this was not tested by the post-hoc lossless transforms.

## DEAD-ENDS

- Current-embedding AR1: +167 B; the older PR130 trial also lost +212 B.
- AR2: +583 B.
- Current pose/carrier features: +271 B with 0/104 fitted weights nonzero.
- Previous partition histograms: +259 B; joint state: +355 B.
- Unchanged RC64 and all four raw-LZMA1 forms lost at the complete-container boundary.

Own-vehicle frontier: **S=0.16959899569230852 @ 187,226 B `[contest-CUDA T4, adjudicated, n600]` (lc2), unchanged by HP4.**