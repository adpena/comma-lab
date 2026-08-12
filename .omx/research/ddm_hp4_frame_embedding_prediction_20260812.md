# DDM HP4 — causal frame-embedding prediction

Date: 2026-08-12  
Axis: `[macOS-CPU advisory, scorer-free byte-only]`  
Verdict scope: `FORMULATION` — exact post-hoc lossless prediction of the pinned CP135 HP3-step2 600x8 int4 embedding by AR1, AR2, current counted-carrier integer features, previous decoded-partition histograms, or their fitted joint form.  
Score claim: `false`

## MEASURED RESULT

| Complete-container treatment | Best real coder | Embedding payload B | Charged predictor B | `archive.zip` B | Delta vs CP135 B | Derived rate-only delta S | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| CP135 incumbent | joint HPAC Brotli q10 | inseparable | 0 | 186,252 | 0 | 0 | baseline |
| Zeroed-field diagnostic | joint HPAC Brotli q10 | inseparable | 0 | 185,193 | -1,059 | -0.000705 | conditional price only; not a valid candidate |
| Order-0 split/repack | Brotli q11 | 1,055 | 0 | **186,247** | **-5** | **-0.000003329** | real byte win; no state prediction |
| AR1 residual | Brotli q11 | 1,227 | 0 | 186,419 | +167 | +0.000111198 | closed |
| AR2 residual | RC64 adaptive base4 | 1,643 | 0 | 186,835 | +583 | +0.000388196 | closed |
| Previous partition histograms | Brotli q11 | 1,231 | 88 | 186,511 | +259 | +0.000172457 | closed |
| Counted pose/carrier row | Brotli q11 | 1,227 | 104 | 186,523 | +271 | +0.000180447 | closed |
| Joint pose + partition state | Brotli q11 | 1,231 | 184 | 186,607 | +355 | +0.000236380 | closed |

The charter hypothesis is **CLOSED_COMPLETE_CONTAINER_DELTA_POSITIVE** in the stated formulation. Every candidate that actually predicts from prior or already-decoded state made the complete archive larger; the best predictive row is AR1 at **+167 B**. The sole win is an order-0 container repack at **-5 B**, so it is not evidence that the embedding is predictable from decoded state. The runner's aggregate `SURVIVES_COMPLETE_CONTAINER_BYTE_WIN` label includes order-0; the independent custody receipt narrows the adjudication correctly.

No derived score delta is promoted: the byte result is authoritative, and any score conversion must be recomputed from the exact denominator before reuse. No scorer row was run. The exact contest and own-vehicle pointers are unchanged; this unit did not make goal progress toward sub-0.15.

## PINNED OBJECT AND CURRENT PRICE

- CP135 archive: `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/candidates/hp3_step2/split_brotli_per_section_opt_cap1_metadata__rc64/archive.zip`; 186,252 B; SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- `models.bin`: 70,825 B; SHA-256 `a8cfe80f24f40ad05c4dcf75ae5fa34b73b8b3412400c5cf8476f5bcb2e4daef`. Its 13,910 B outer HPAC stream jointly codes every HPAC field, so the embedding has no honest separable incumbent compressed span.
- Located IHS2 frame embedding: HPAC-body offset 13,379; shape 600x8; signed int4; 2,400 physical bytes; retained SHA-256 `ded285fd1f9f1df2810fbdfe50e34c5d81a40cc5817a4a365b20998b1942c4c8`.
- Real current-container conditional price: zeroing only the field and rebuilding the unchanged incumbent container yields 185,193 B, hence 1,059 B. This is a diagnostic counterfactual, not a semantically valid output.
- Counted carrier coefficients used as already-decoded state: `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/pr135/decoded/coefficients_signed_int12.int16.npy`; 14,528 B; SHA-256 `005f4ddc9cbb9718619a18b87665dce6592035051746142de8b8e04326fc0fe7`.
- Decoded partition state: `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/coders/hp3_step2/decoded_spatial_tokens.fresh_rc64.bin`; 117,964,800 B; SHA-256 `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.

## REAL CODER RACE

All rows rebuilt the full ZIP; no isolated-stream delta was used as the verdict.

| Order-0 codec | Complete-container delta B |
|---|---:|
| Brotli q11 | **-5** |
| RC64 adaptive base4 | +87 |
| LZMA1 raw cp64k | +264 |
| LZMA1 raw pr101-4k | +272 |
| LZMA1 raw hp1-16m | +272 |
| LZMA1 raw e4-1m | +317 |

The exact F26 ExperimentBook RC64 implementation was used, pinned by Python SHA-256 `f5826e1a871971451f6a79e482c30c425bd8bc3ccde652196fe8320222593aba` and C SHA-256 `5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6`; Brotli was 1.2.0. Order-0 residual H0 was 7,926.04 bits, versus 9,239.57 for AR1 and 12,563.78 for AR2. AR1 and AR2 therefore worsen the residual statistics before container overhead. The fitted pose-only predictor used 0/104 nonzero parameter bytes; partition and joint used 13/88 and 13/184, respectively.

## RECEIVER, DETERMINISM, AND PAYLOAD CUSTODY

- Denominator: 6 predictors x 6 real coder forms = 36 complete-container candidates.
- Every candidate decoded its residual causally and reconstructed the exact IHS2 bytes; semantic, carrier, residual, and token streams remained byte-identical.
- All 36 candidate archives, models, members, coder payloads, parameters, raw residuals, and deterministic repeats were retained. All repeat hashes matched.
- Independent audit verified all 396 candidate artifact records, every repeated archive/model/coder payload, and every ZIP member. The final immutable manifest covers 485 retained files.
- Winner archive: `/Volumes/VertigoDataTier/pact/ddm_hp4/retained/candidates/order0/brotli_q11/archive.zip`; 186,247 B; SHA-256 `3d80f8ab16212abcbec213bf76036c2ca785284cbe924c092cca04549d00f7cc`.
- `FINAL_RESULT.json`: `/Volumes/VertigoDataTier/pact/ddm_hp4/retained/FINAL_RESULT.json`; 139,312 B; SHA-256 `f7852464abb3f1a88842393e6773738410cc26d332cb68af18a9a23a1d0cbee1`.
- Independent custody receipt: `/Volumes/VertigoDataTier/pact/ddm_hp4/retained/FINAL_CUSTODY_RECEIPT.json`; 6,516 B; SHA-256 `670d36228bbd777dfba0ea67757d69007ebab446938afc5830fcccbd4f25ddb4`.
- Final immutable manifest: `/Volumes/VertigoDataTier/pact/ddm_hp4/retained/FINAL_IMMUTABLE_MANIFEST.json`; 151,624 B; SHA-256 `cf1a3cf7da9b3907a5a971ade77c8e5f959926525c6d8372fc230ca96a7ef062`.
- Executed runner: `/Volumes/VertigoDataTier/pact/ddm_hp4/retained/run_hp4_frame_embedding_prediction.py`; SHA-256 `0f1b25d6ee7bd711a960065e8bba8732e4a851d4505978a1dfd55dee5f43e717`. Independent auditor SHA-256: `e5acb538a2178738c054010cf792174066802ec20c824ed5251341a0752626c8`.

The first `ARTIFACT_MANIFEST.json` is retained but superseded only for `run/main.safe_run.json`: the wrapper finalized that mutable telemetry receipt after the child wrote the first manifest. No candidate or measurement artifact mismatched. `FINAL_IMMUTABLE_MANIFEST.json` excludes mutable run telemetry and is the custody authority.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus and receipts by content for `frame_embed`, `IHS2`, `600x8`, `HP31`, `temporal delta`, `decoded partition`, `carrier coefficients`, `RC64`, `Brotli`, `LZMA1`, and `complete container`; searched canonical equations for entropy, residual, causal, rate, and container surfaces; searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC documents, the task ledger, and `.omx/state/main_hot_state.md`.

Findings beyond the charter seeds changed the plan:

- `ddm_hp3_20260810/FINAL_REPORT.md` had already tested exact AR1 modulo prediction on the older PR130 object and lost **+212 B** at the full-container boundary. HP4 therefore pinned the newer CP135 HP3-step2 object and treated AR1 as a control, not as an untested invention.
- `ddm_fd135_fractal_decomposition_20260810.md` supplied the exact IHS2 map: the 600x8 int4 field is 2,400 raw bytes at HPAC-body offset 13,379 inside a 16,593 B body. This removed any need for heuristic section discovery.
- `ddm_cp135_rate_compose_20260810.md` and `ddm_pi135_pr135_intake_20260810.md` pinned the current CP135 complete-container object and its real joint-Brotli boundary. The experiment therefore rebuilt every archive rather than comparing isolated payloads.
- `ddm_eu3_fresh_eyes_eureka_20260812.md` ranked the state-prediction idea but reported no complete-container measurement. HP4 supplied that missing test.
- The canonical equations registry did not displace the required exact residual-decode and complete-container protocol; no cheaper already-measured representation of this exact current embedding was found in the searched index/DAG/ledger scope.

## BOUNDARIES

- No scorer, renderer, `evaluate.py`, Modal, MPS, CUDA, or network operation ran.
- This is a scorer-free lossless byte result. The derived score deltas assume identical distortion and cannot promote a contest or own-vehicle pointer.
- The zero-field row is only a conditional marginal measurement; it is not valid output.
- The receiver seam reconstructs the exact original HPAC body. Production F26 integration was unnecessary to close the post-hoc state-prediction formulation.
- This does not kill jointly retraining the probability object so that its learned embedding is prediction-friendly; that is a different family and would need a new charged, byte-closed archive.

Own-vehicle frontier: **S=0.16959899569230852 @ 187,226 B `[contest-CUDA T4, adjudicated, n600]` (lc2), unchanged by HP4.**

## NEXT_IF_RESUMED

- `order0_brotli_q11_repack` — disposition: `FOLDED`; owner: `MAIN / next CP135 probability-object composer`; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/`; fire trigger: the next score-bearing probability-object archive is being built and its receiver can absorb HP4M without a dedicated scorer row. Recount against that exact parent and integrate only if complete receiver parse-back remains exact and the full-container delta stays negative.

## LIVE-HYPOTHESES

- The -5 B order-0 split may survive or be absorbed into the next score-bearing probability-object composition because its retained q11 payload plus the repacked remainder beat the pinned parent at the exact ZIP boundary; the gain is small enough that a changed parent may erase it.
- A jointly retrained state-conditioned HPAC probability object may reduce total model-plus-token bytes even though post-hoc prediction failed: training could change the learned representation itself, whereas current AR1 raises residual H0 and the tested pose fit collapses to zero weights. This is an untested family, not a reason to retry the closed post-hoc transforms.

## DEAD-ENDS

- Exact post-hoc AR1 on the current CP135 embedding: +167 B complete-container; the older PR130 test also lost +212 B.
- Exact post-hoc AR2 on the current embedding: +583 B at its best real coder.
- Current counted pose/carrier features: +271 B; the fitted predictor had 0/104 nonzero parameter bytes.
- Previous decoded-partition histograms: +259 B; joint pose-plus-partition state: +355 B.
- RC64 on this current order-0 representation: +87 B versus Brotli q11 at -5 B; do not rerun it unchanged.
- All four tested raw-LZMA1 forms lost at the complete-container boundary (+264 to +317 B); do not rerun them unchanged.
