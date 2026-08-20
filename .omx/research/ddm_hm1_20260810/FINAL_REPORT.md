# ddm_hm1 HPAC model-capacity result

## Outcome

**FORMULATION verdict:** post-hoc coordinate removal from the shipped PR130 D8 frame-conditioning vector is at a one-coordinate knee and does not improve the full n600 joint rate. The seeded stratified-random n120 screen selected `frame_dim7_drop5`, but its byte-closed n600 archive is **191,112 B**, which is **+60 B** versus the pinned 191,052 B PR130 archive.

This is a scorer-free rate result on `[macOS-CPU advisory; scorer-free real serialized bytes]`. No `upstream/evaluate.py` job ran, no scorer slot was claimed, and the frontier did not move.

## Measured joint curve

Selection mode: seed `20260810`; 10 temporal strata x 12 randomly selected frames = n120; never a prefix. Every cell retained its checkpoint, HPAC raw/xz payload, code/logit materialization, symbols, real Range payload, and decoded payload with byte count and SHA-256.

| Cell | Dropped coordinates | HPAC xz B | n120 Range B | Projected n600 joint B | Delta vs D8 projection B |
|---|---:|---:|---:|---:|---:|
| D8 control | none | 15,164 | 23,376 | 132,044 | 0 |
| D7 drop0 | 0 | 14,708 | 23,476 | 132,088 | +44 |
| D7 drop1 | 1 | 14,748 | 23,452 | 132,008 | -36 |
| D7 drop2 | 2 | 14,764 | 23,464 | 132,084 | +40 |
| D7 drop3 | 3 | 14,744 | 23,468 | 132,084 | +40 |
| D7 drop4 | 4 | 14,788 | 23,452 | 132,048 | +4 |
| **D7 drop5** | **5** | **14,744** | **23,452** | **132,004** | **-40** |
| D7 drop6 | 6 | 14,724 | 23,492 | 132,184 | +140 |
| D7 drop7 | 7 | 14,820 | 23,464 | 132,140 | +96 |
| D6 drop0,5 | 0,5 | 14,436 | 23,568 | 132,276 | +232 |
| D6 drop1,5 | 1,5 | 14,424 | 23,520 | 132,024 | -20 |
| D6 drop2,5 | 2,5 | 14,380 | 23,536 | 132,060 | +16 |
| D6 drop3,5 | 3,5 | 14,428 | 23,552 | 132,188 | +144 |
| D6 drop4,5 | 4,5 | 14,468 | 23,528 | 132,108 | +64 |
| D6 drop5,6 | 5,6 | 14,392 | 23,580 | 132,292 | +248 |
| D6 drop5,7 | 5,7 | 14,436 | 23,540 | 132,136 | +92 |

No D6 child beats its D7 drop5 parent. The n120 projection chose D7 drop5 at -40 B versus the projected D8 control, but the full n600 recode reversed the sign:

| n600 component | PR130 base B | D7 drop5 B | Delta B |
|---|---:|---:|---:|
| HPAC model xz | 15,164 | 14,744 | -420 |
| Real Range token payload | 116,980 | 117,464 | +484 |
| **Model + token joint** | **132,144** | **132,208** | **+64** |
| **Exact archive.zip** | **191,052** | **191,112** | **+60** |

The selected archive SHA-256 is `ffaf7cbeb3e8211c2f8c9cf9f643c7faf1e0d5bd26de2a453d8630968e92ff1d`. The retained resumable token payload is 117,488 B with SHA-256 `a2906df405be674504f52defb52e38e7ea1e20c146260f8590367394d2408627`; full and seek decode are exact.

## Capacity verdict and falsifier

The shipped D8 model is **locally saturated for post-hoc coordinate deletion**: one coordinate can be removed, but its 420 B model saving is overpaid by 484 B of token entropy; deleting any second coordinate from that winner is already worse on the n120 joint curve.

This verdict is FORMULATION-scoped. It does not kill trained growth, trained pruning, width/depth changes, or new quantization. The direct falsifier is a trained, retained HPAC cell whose full n600 real-coder joint is below 132,144 B, followed by an exact archive below 191,052 B with byte-identical receiver output. The existing CL1 trained growth ladder remains queued because this sandbox has no MPS device and `ddm_sd2` owns the active local-Metal lane.

## Byte-closed receiver and $0 parity

The public staged receiver decoded the exact archive payload, retained decode checkpoints at 300/600 frames and render checkpoints every 24 frames, and produced a 3,662,409,600 B raw file with SHA-256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`, exactly matching the pinned PR130 raw.

| Parity half | Frames | Denominator bytes | Changed bytes | Max abs |
|---|---:|---:|---:|---:|
| Even pose carrier | 600 | 1,831,204,800 | 0 | 0 |
| Odd semantic | 600 | 1,831,204,800 | 0 | 0 |

The successful resume-only receiver invocation took 0.74 s because it consumed the already-complete checkpoints. The original materializing receiver run completed all stages before its log-retention guard stopped receipt writing; decode wall time is report-only and is not an admissibility claim.

## Free-side audit

Already-free generic receiver objects are patch-group masks and sparse gather plans, coordinate grids and causal scan order, integer arithmetic, and the Range decoder algorithm.

Counted video-derived objects remain all learned HPAC weights, biases, exponents and per-row depths, the 600 x D frame embeddings, and the Range-coded semantic tokens. No new derive-instead-of-store win was proven on the exact shipped model. Under IHS1, per-row bit depths are needed to parse the variable-bit weight stream before weights exist, so they are not decoder-derivable without a new self-delimiting representation.

## RECALL EVIDENCE

Queries covered `HPAC`, `frame embedding`, `capacity`, `model + token`, `IHS1`, `per-row depths`, and `self-delimiting` across `.omx/research/`, arm final messages, source, experiments, tools, the canonical-equations listing, the canonical research index/DAG surfaces, and task-ledger material.

Beyond the charter seeds, the recall found:

- `.omx/research/ddm_cl1_capacity_20260809/{PREREGISTRATION.md,MAIN_METAL_FIRE_ORDER.md,BLOCKED_RECEIPT.md}`: a trained capacity ladder already exists and has a governed Metal fire order. This prevented duplicating training in the sandbox and made the post-hoc curve the immediate measurable branch.
- `.omx/research/ddm_rr1_20260809/RECALL_AUDIT.md` and `.omx/research/ddm_tm1_20260809/TM1_FINDINGS.md`: receiver/coder custody and token-model coupling must be priced through real recodes. This made every selection cell retain real Range and decoded payloads.
- `.omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md` and `.omx/research/ddm_op1r_20260809/OP1R_PATH.md`: scorer work is downstream of a byte win, not a substitute for one. This kept the arm scorer-free after the +60 B archive result.
- `src/tac/pr130_runtime/fx1_runtime_tree/integer_model_io.py` plus HP3 runtime sources: IHS1 parse order consumes per-row depths before weights are available. This closed the naive derive-depths-for-free idea on the current representation.

## Provenance and custody

Pinned inputs reproduced:

- Base archive: 191,052 B, SHA-256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
- Base raw: 3,662,409,600 B, SHA-256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`.
- Source checkpoint: SHA-256 `0f4775920aeb2fb419555cc4d68703dd90b88be9d24c82466a99fddc1b1f1aa7`.
- Canonical cache: SHA-256 `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`.
- PR130/HP3 code reuse: `experiments/ddm_hp3_hpac_section_and_zip_frame.py`, `src/tac/pr130_runtime/ddm_hp3_runtime/`, and `src/tac/pr130_runtime/fx1_runtime_tree/` at parent `7e4f7a8c38`.

Durable source commits live in the validated sparse clone at `/Volumes/VertigoDataTier/pact/ddm_hm1_20260810/source_repo_sparse`:

- `af8d22906e` — retained HPAC frame-dimension curve and tests.
- `3ebc1f4f28` — retained per-cell coder resume.
- `b9258c9fcf` — retained n600 stage resume.
- `6157f729b5` and `0640ae23b7` — artifact-gated closure resume.

The primary checkout has matching source files but cannot accept Git object writes in this sandbox; unrelated dirty work and the shared index were not touched. A failed full clone remains retained at its source, and two non-authoritative partial APDataStore copies remain after cross-volume extended-attribute moves were blocked. No clone bytes were deleted; `failed_clone_cleanup.json` records the bounded cleanup disposition.

## Validation

- `pytest -q experiments/ddm_hm1_20260810/test_hm1_frame_dim_curve.py`: 2 passed.
- Ruff and Python compilation passed for the arm sources.
- Python files received two review-tracker passes after their final edits.
- Payload-retention preflight found no measure-and-discard violation in the arm sources.
- Archive ZIP integrity, full/seek Range decode, decoded-token identity, receiver raw identity, and both parity denominators passed.
- No upstream file was modified.

## Dispositions

- **CLOSED-NO-WIN:** post-hoc D8→D7→D6 coordinate deletion on the shipped model. Consumer: this report and the machine receipt.
- **NOT-QUEUED:** exact scorer evaluation. Fire condition failed because the archive is +60 B, not smaller.
- **QUEUED-WITH-EXISTING-FIRE-ORDER:** CL1 trained HPAC growth/pruning ladder. Owner: `ddm_cl1_capacity MAIN unsandboxed Metal executor`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`; fire trigger: `ddm_sd2` releases the local-Metal lane and the CL1 ticket's existing guard passes.
- **BLOCKED-RETAINED:** failed-clone cold-store cleanup. Owner: operator with filesystem authority; consumer store: `/Volumes/APDataStore/pact/ddm_hm1_failed_source_repo_20260810*`; fire trigger: a copy method that can preserve or explicitly certify extended-attribute loss, followed by verified source/destination identity.

## Own-vehicle frontier

Unchanged: S = 0.7539807296911207 at 357,836 B `[macOS-CPU advisory] n600`. This arm produced no score and did not move the contest pointer.

## LIVE-HYPOTHESES

- Trained capacity reallocation may beat post-hoc deletion because optimization can redistribute predictive load before paying the token penalty; CL1 already contains the governed ladder needed to test this.
- A self-delimiting IHS successor could remove or compress the per-row-depth side information because the current no-go is caused by parse order, not an information-theoretic proof that depths must be stored.
- The n120 rank reversal suggests future capacity selection needs either a finite-state correction calibrated on controls or a larger stratified sample before committing an expensive n600 winner.

## DEAD-ENDS

- Post-hoc removal of coordinate 5 is closed as a rate win: model -420 B, tokens +484 B, joint +64 B, archive +60 B.
- Removing any second coordinate from the D7 drop5 winner is closed at the n120 formulation screen: all seven D6 children are worse than their D7 parent.
- Deriving IHS1 per-row depths from already-decoded weights is closed on the current wire format because those depths are required before the weights can be parsed.
- Exact scoring this archive is closed: it already loses the byte gate and is byte-identical to the base raw, so scoring cannot improve its total.
