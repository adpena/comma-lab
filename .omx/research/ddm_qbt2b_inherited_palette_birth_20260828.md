# ddm_qbt2b — inherited FP1 palette, data-dependent QBFLOW readout fit, and CE-birth r3 fire order

## OUTCOME

This arm built and measured the corrected initialization, extended the existing QBT1 trainer with the real realized-CE birth stage, proved bounded checkpoint/archive identity across the 03a/03 boundary, corrected the AP on-disk storage projection, and sealed an unlaunched r3 fire order for MAIN.

| surface | result | disposition |
|---|---:|---|
| readout design rank | **65/65** | nondegenerate |
| fit residual / palette-target variance | **0.2851456635** | below the 0.95 degeneracy gate |
| init classes present with within-class error <60% | **1/5** | INIT GATE MISS; CE remains live by charter |
| before → after unweighted n32 pose MSE | **0.0830440460 → 132.286409259** | severe pose collateral measured |
| one-step n1 CE smoke | CE **2.445890188**, pose MSE **160.259002686** | mechanism only; no n32 verdict |
| 03a/03 checkpoint + archive identity | **PASS**, archive 107,501 B, sha `42199821…` before/after reload | bit-faithful |
| projected peak memory | **92,084,098,826 B** / 124,554,051,584 B | PASS, 32,469,952,758 B headroom |
| corrected retained-output demand | **25,773,400,502 B** | projection only |
| live-free requirement including 10% safety + 8 GiB floor | **36,940,675,145 B** vs **65,683,980,288 B** live | PASS, 28,743,305,143 B headroom |
| training / Metal / Modal / contest eval | **0 / 0 / 0 / 0** | arm did not launch |
| frontier movement | **none** | goal not achieved in this arm |

All measurements are `[macOS-CPU frozen-scorer advisory]` or explicitly scorer-free, `score_claim=false`. No n600 scorer job and no contest evaluation ran.

## PALETTE CUSTODY

The inherited palette is **VIDEO-DERIVED, CE-TRAINED, INHERITED**. It is the retained FP1 `proto_solved` payload from the documented 32-pair/100-step full-scorer CE solve, not a head-only or analytic construction.

- Source: `/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/prototypes.npz`
- Bytes: 1,158
- SHA-256: `19e6524b75724f0b19f0e2e49a827d9f28b40d087b1e5504c3a85577a9e76f0b`
- Exact AP copy: `initialization/palette/inherited_fp1_prototypes_exact.npz`, byte-identical SHA
- Values payload: `initialization/palette/inherited_fp1_palette_values.npz`, 886 B, sha `e6cc27d6…`

| class | inherited RGB |
|---|---:|
| Road | (30.243652, 38.815331, 72.386734) |
| Lane | (77.427879, 86.707779, 118.528511) |
| Undrivable | (157.146561, 65.717026, 56.106434) |
| Movable | (75.504166, 108.055252, 140.977509) |
| MyCar | (129.259796, 153.586746, 152.828293) |

The source FP1 sample IDs are retained in the values payload and initialization receipt.

## DATA-DEPENDENT READOUT FIT

Base state: QBT1 r1 EMA checkpoint, 1,980,573 B, sha `7e3fb97199eca5cd03eac8c2b858bdcea3b6ddad83eba9c410161252455084cd`.

The fit sampled observed r1 `render_state` values deterministically by pair and native class, then regressed last-frame RGB logits onto the inherited class palette. Only `render_out_w[:,3:6]` and `render_out_b[3:6]` changed. Frame-0 columns, every other tensor, all shapes, the QBF1 packet schema, coder, and receiver stayed unchanged.

- Raw fit sample payload: `initialization/fit/readout_fit_samples.npz`
- Samples + predictions + coefficients: `initialization/fit/readout_fit_samples_and_predictions.npz`
- Receipt: `initialization/fit/READOUT_FIT_RECEIPT.json`, 5,962 B, sha `f3560023…`
- Initialized state: `initialization/initialized/initialized_r3_state.pt`, 399,007 B, sha `0bedbd66…`

| native class | sampled states | mean rendered RGB after fit | RGB RMSE by channel |
|---|---:|---:|---:|
| Road | 1,740 | (81.04, 76.49, 94.70) | (53.58, 41.14, 27.09) |
| Lane | 1,152 | (94.80, 85.03, 99.83) | (24.51, 19.30, 26.46) |
| Undrivable | 16,384 | (149.97, 71.82, 64.30) | (18.76, 17.24, 18.95) |
| Movable | 5 | (150.57, 59.06, 52.50) | (75.07, 49.23, 88.58) |
| MyCar | 16,384 | (126.77, 141.71, 142.04) | (15.19, 22.79, 21.27) |

The fit is algebraically nondegenerate, but the five native regions are badly imbalanced: only five Movable states existed in the sampled r1 native partition. The realized scorer receipt, not the regression residual, is therefore the governing initialization result.

## N32 INITIALIZATION RECEIPT

The before side consumes QBT1 r1's already-retained exact camera/logit/argmax/pose bank. The after side rerendered the same seeded-stratified 32 pairs through `R → uint8 → frozen CPU SegNet/PoseNet` and retained one NPZ per pair with the camera frames, logits, argmax, pose, and targets.

| class | target share | before predicted share | before within error | after predicted share | after within error |
|---|---:|---:|---:|---:|---:|
| Road | 23.1196% | 0% | 100% | 0% | 100% |
| Lane | 0.5961% | 0% | 100% | 0% | 100% |
| Undrivable | 49.5828% | 59.4170% | 0.0024% | 100% | 0% |
| Movable | 1.2456% | 0% | 100% | 0% | 100% |
| MyCar | 25.4559% | 40.5830% | 0.0555% | 0% | 100% |

The inherited initialization collapsed the realized output to Undrivable only. It passed **1/5**, not the required 4/5, and worsened unweighted n32 pose MSE from 0.0830440460 to 132.286409259. This closes the inherited readout as a zero-step birth mechanism at INSTANCE scope. It does not close the CE-birth curriculum because FP1's palette itself emerged only after CE training and the charter explicitly makes an init-gate miss nonblocking.

Primary receipt: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/qbt2b_inherited_palette_birth/initialization/INITIALIZATION_RESULT.json`, 32,522 B, sha `1b8dba7a…`.

## TRAINER EXTENSION

Changed files:

- `experiments/ddm_qbt1_qbflow_trainer.py`, sha `6f547de062f506d277339f4404dbd36281e3267351d1e9eb96a0cfdef7cacc06`
- `experiments/tests/test_ddm_qbt1_qbflow_trainer.py`, sha `9cb7e72ae9816713cee80eaf5a0370e5a4f062abfc64dd68cbb0c4d00d27078c`

Implemented behavior:

1. `stage_03a_ce_class_birth` optimizes per-pixel CE from the realized frozen-SegNet logits on the real render/R/uint8 path.
2. Pose MSE and its score contribution are active from the first CE update.
3. The handoff threshold is derived as `1 - DEFAULT_TAU_PERSIST = 0.20`; all five classes must be present and below that error for two consecutive realized verdicts.
4. A 100-step safety cap stops without entering margin training if the event never fires. It is not a step-triggered handoff.
5. The unchanged joint expected-flip margin+pose stage follows only after the event.
6. Additive config fields preserve legacy validation. EMA is resolved by `resolve_ema_law(5100)` for the maximum 100+5,000-update schedule. The hard chunk ceiling remains 30; governed n32 training remains two equal-mass 16-pair chunks.
7. Checkpoints retain the phase, birth cursor, margin cursor, verdict count, consecutive-pass count, optimizer, RNG, live state, and EMA. Periodic history stores compact SHA-bound re-encode summaries so checkpoint history does not recursively embed coder inventories.
8. Each QBT2B checkpoint re-encode retains all real coder candidates in one verified deterministic tar plus one manifest. The scored archive itself repeats byte-identically.

The QBF1 packet ABI was not edited.

## BOUNDED SMOKE

Final receipt: `birth_smoke_n1_final_v3/RESULT.json`, 59,265 B, sha `9ad23f7d…`.

- Scope: n1, one real CPU CE update, no authority claim.
- CE: 2.445890188; pose MSE: 160.259002686; total: 284.621398926.
- Realized verdict: only Undrivable present; pose MSE 158.001655434.
- Handoff: correctly refused; margin stage did not run.
- Checkpoint cursor after reload: birth step 1, verdict count 1, stable count 0, margin step 0, handoff false.
- Archive before/after reload: 107,501 B, sha `4219982128e55d385b55289fcb48afb299f47d5c0ed0a886e4af2f5e3663e60b`.
- Consolidated tar before/after reload: sha `a0b30022…` both times.
- Every verdict camera frame, scorer logit, argmax, pose, and target is retained.

This smoke proves the CE/scorer/checkpoint/re-encode mechanism and the 03a/03 serialization boundary. It does not measure n32 birth, a two-verdict event, Metal speed/memory, n600 distortion, or contest score.

## STORAGE LEG 2

The first review pass found that a constant checkpoint-size model was false: retained R2 checkpoints grew from 1,603,805 B at step 5 to 15,818,589 B at step 4,865, or 2,924.8527 B/update. The final projection conservatively applies the R2-derived 16,505,930 B worst-case final checkpoint size to every one of 1,020 periodic checkpoints even though QBT2B now stores compact re-encode history.

The binding periodic formula is:

`(18,604,053 logical B + 6 files × 131,072 B/cluster) × 1,020 = 19,778,294,700 B`

Additional retained demand includes 20 maximum n32 birth verdict payloads (5,423,153,920 B), 38 stage/precision re-encodes plus four stage checkpoints (166,584,650 B), the final n32 evaluation (271,149,504 B), and 128 MiB metadata reserve.

Final projection receipt: `sealed_r3/STORAGE_PROJECTION.json`, 4,523 B, sha `99183f29…`.

## SEALED R3 FIRE ORDER

- Fire order: `sealed_r3/SEALED_R3_FIRE_ORDER.json`, 11,150 B, sha `8ae8fcf2…`
- Draft config: `sealed_r3/COMPILED_N32_R3_CONFIG.json`, 6,302 B, sha `4541f5a9…`
- Draft validation: `sealed_r3/DRAFT_CONFIG_VALIDATION.json`, 600 B, sha `022f3e02…`
- Review receipt: `TWO_PASS_REVIEW_RECEIPT.json`, 1,108 B, sha `70154da0…`
- Disposition: `QUEUED_R3_STAGE03A_03_04_FIRE_STAGE05_BLOCKED`.

The config is deliberately sealed with `launch_authorized=false` and empty Metal/scorer claims. Draft validation passes when authority is not required and refuses launch with `heavy training is not authorized`. MAIN must copy it, re-read AP free space and memory admission, verify the committed hashes, confirm there is no duplicate active scorer/Metal lane, bind real claim IDs, and only then authorize stage 03a.

Stage 03 may begin only after two consecutive all-five-class realized verdicts below 20% within-class error. Stage 05 remains blocked until a real retained same-budget QBW1 control passes the unchanged custody, pair-set, byte-budget, and arithmetic checks.

The CPU-smoke linear walltime projection is 742,607.668 s (206.28 h) for the maximum 5,100-update schedule. This is deliberately conservative and is not a Metal measurement; MAIN must remeasure on Metal before fire.

## RECALL EVIDENCE

Searched before build:

- Corpus content query: `inherited palette|class-birth|CE births|margin sharpens|render_out_w|readout fit|palette.*QBFLOW|rare-class-protected|interior-conflict` across `.omx/research`, `.omx/state`, `docs`, `experiments`, and `src`.
- Event-law query: `CE.*birth|birth.*CE|margin.*sharpen|event-trigger|within-class error|all five classes|rare class` across research memos.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, then class/margin/birth/palette/QBFLOW/pose filters.
- Research graph/index/task surfaces: `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, canonical task-status ledger, and live hot state.
- Primary implementations: FP1 `solve_prototypes`, QBT1 trainer/packet twin, `birth_completion.py`, and the retained r1/r2 checkpoint/re-encode corpus.

Beyond the charter seeds, the recall found the canonical Morse-Smale completion threshold (`DEFAULT_TAU_PERSIST=0.8`), the broader event-triggered `island_birth → boundary-form → tau-sharpen` law, the p4x matched-pose-collateral requirement, and the measured R2 checkpoint-growth curve. These changed the build by deriving the 20% handoff threshold from code, keeping pose active in every birth update/verdict, and replacing the constant-size storage model with a worst-case growth-aware one. It did not find a separate existing QBFLOW palette/readout solution in the searched scope.

## TESTS AND REVIEW

- `.venv/bin/python -m pytest experiments/tests/test_ddm_qbt1_qbflow_trainer.py -q` → **13 passed**.
- `.venv/bin/ruff check experiments/ddm_qbt1_qbflow_trainer.py experiments/tests/test_ddm_qbt1_qbflow_trainer.py` → **all checks passed**.
- `git diff --check` → clean.
- Two genuine visible review passes covered 73 trainer entities and 14 test entities per pass. No Python review override was used.
- Independent audit reloaded the final checkpoint, verified the exact curriculum cursor, recomputed class-share sums and init-gate count, rehashed the palette/state/checkpoint payloads, and confirmed archive and tar identity across reload.
- `upstream/` and `experiments/ddm_qbflow_packet.py` have no diff. The pre-existing staged index remained untouched and empty.

A first pre-receipt fit invocation failed before the raw design was written and is inadmissible as evidence. The final implementation moves raw-sample persistence before the solve; the same deterministic selection was replayed and both the raw and derived fit payloads are retained. No result in this memo comes from the failed invocation.

## BOUNDARIES

Measured: inherited palette custody; observed-render-state regression; n32 before/after realized scorer table; pose collateral; n1 CE mechanism; checkpoint/RNG/EMA/curriculum-state reload; QBF1 archive repeat; cluster-aware storage projection; live AP free space at seal time.

Not measured: a governed n32 CE-birth result, successful two-verdict handoff, any margin descent after birth, Metal walltime/peak, n600 Seg/Pose, exact contest CPU/CUDA score, same-budget QBW1 stage-05 control, or frontier movement.

## LIVE-HYPOTHESES

- Realized CE can still birth all five classes within the 100-step n32 window because the inherited FP1 colors themselves required CE training; the zero-step transfer failure does not test the training mechanism.
- Pose-active CE may recover the 132.286 initialization collateral while classes are born because QBT1 previously showed the same vehicle's pose path can descend strongly through realization; the joint n32 trajectory is unmeasured.
- The readout residual ratio of 0.285 with full rank suggests the renderer state contains some class-separating signal, but the five-pixel Movable native support means CE must first expand rare-class support before the palette can become useful.
- If the two-verdict birth event fires, the unchanged margin law may finally move realized d_seg below the 0.2504 QBT1 floor because it will begin from a five-class partition instead of the proven two-class basin.

## DEAD-ENDS

- FP1 palette relabeling as a non-trained construction remains closed: its primary implementation is a 32-pair/100-step full-scorer Adam/CE solve.
- The rank-4 feature quotient is not an RGB inverse and cannot initialize this renderer.
- A head-only solve cannot stand in for the nonlinear SegNet body on the realized RGB path.
- QBT1 margin-from-step-zero is closed at INSTANCE scope after 4,670 flat realized-seg steps.
- The inherited readout fit alone is closed as a zero-step class-birth mechanism at INSTANCE scope: it produced only Undrivable and passed 1/5 init classes.
- Treating unchanged frame-0 RGB columns as pose protection is closed: PoseNet reads both frames, and last-frame recoloring raised unweighted n32 pose MSE to 132.286.
- Constant checkpoint-size storage projection is closed: R2 measured 1.60 MB → 15.82 MB growth, so the sealed projection uses the worst-case final size.
- A one-step n1 CE smoke is not a birth verdict and must not be retried as one; its only valid use is mechanism and resume-boundary proof.

## NEXT_IF_RESUMED

- **QUEUED** — owner: MAIN QBFLOW r3 inherited-palette CE-birth owner; consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r3`; fire trigger: committed hashes and two-pass receipt verified, live AP and ≤116 GiB admission rechecked, no duplicate scorer/Metal lane active, both real lane IDs bound, and an authorized copy of the sealed config created.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN QBFLOW r3 stage-05 owner; consumer store: `governed_n32_r3/stage_05_same_budget_admission`; fire trigger: r3 reaches stage 05 and a real retained same-budget QBW1 control passes custody, pair-set, byte-budget, and score-arithmetic validation.

gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]
