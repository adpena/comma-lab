# DDM-SM4 low-rank mechanism discriminator

## Verdict

**Rank insufficiency, not double-int4 factor quantization, is the dominant failure mechanism at the recovered-byte budget.** The real 410-cell PR130 grid found that r16-int8 raises aggregate relative weight error from **0.50672 to 0.69459** versus r32-int4, a **1.371x regression**, and loses on **5/5 tensors**. This directly falsifies the charter's factor-precision rescue at the equal raw-factor byte law.

Centering is real but too small to repay its counted bytes. At fixed r32-int4, stored fp16 row means reduce error by only **0.883%** while adding **968 archive bytes**. Under the actual **182,364 B** matched archive ceiling, the best centered cell is r29-int4 at **0.53606**, which is **5.79% worse** than uncentered r32-int4 at **0.50672**.

The best eligible grid cell is therefore r32-int4 itself. Its staged receiver state is byte-identical to the already-refuted SM3/CP2 r32 state (`5063e24b...`). The prior retained full RAW already proves even/pose frames exact, odd/semantic frames at mean absolute delta **67.71** with **99.26%** of channel values changed, and local exact advisory score **7.4924**. No new full inflate or scorer request was warranted.

**BASE: PR130 CPR1 S = 0.172141297491896447 @ 191,052 B `[contest-CUDA, DALI GT, n600]` — UNMOVED.**

## Matched-byte table

| Cell | Semantic bytes | Actual temporal-composed archive bytes | Aggregate relative L2 | Disposition |
|---|---:|---:|---:|---|
| r32-int4, uncentered | 32,774 | 182,364 | 0.50672 | Best under budget; exact-state match to the settled catastrophic r32 candidate |
| r16-int8, uncentered | 32,614 | 183,312 | 0.69459 | Factor-precision rescue refuted; 1.371x more error |
| r32-int4, centered | 33,734 | 183,332 | 0.50225 | 0.883% less error for +968 archive bytes; dominated |
| r29-int4, centered | 32,258 | 182,036 | 0.53606 | Best centered cell under budget; worse than r32-int4 |
| shipped direct q4 control | 40,252 | 188,636 with temporal tokens | 0.12788 | Positive control; low-rank budget optimum has 3.962x its error |

The r16-int8 semantic field is 160 bytes smaller than r32-int4 under the intended raw-factor law, but its real outer archive is 948 bytes larger because the higher-precision factor stream compresses worse inside the model section. That archive interaction only strengthens the negative; the mechanism verdict already follows from the matched raw-factor law and the 5/5 tensor error direction.

## Per-tensor discriminator

| Tensor | r32-int4 relative L2 | r16-int8 relative L2 | r16/r32 |
|---|---:|---:|---:|
| `blocks.0.pw.weight` | 0.49849 | 0.69630 | 1.397x |
| `blocks.1.pw.weight` | 0.51467 | 0.71338 | 1.386x |
| `blocks.2.pw.weight` | 0.46833 | 0.67192 | 1.435x |
| `blocks.3.pw.weight` | 0.46935 | 0.66897 | 1.425x |
| `coord_mix.weight` | 0.50656 | 0.67212 | 1.327x |

The discriminator's falsifier was explicit: if r16-int8 lowered aggregate error and the same five per-tensor errors, double-int4 compounding would remain viable. It did neither. The verdict scope is the five named matrices with one global rank/precision, optional stored fp16 row means, and actual temporal-composed archives at or below the r32-int4 budget. It does not claim that every possible heterogeneous or residualized matrix codec is dead.

## Receiver and payload custody

- All **410/410** grid cells retained `semantic.bin`, `models.raw`, independent decoded state, and both deterministic `models.xz` / packet / `archive.zip` builds under `/Volumes/VertigoDataTier/pact/ddm_sm4_20260810/retained/grid/`.
- The full grid receipt is `/Volumes/VertigoDataTier/pact/ddm_sm4_20260810/GRID_RESULT.json`, **2,911,574 B**, SHA-256 `49414acc4efc203d9a38bf63775afaa690ae603b4b11628155d6bf76e694ae52`.
- The staged selected archive is `/Volumes/VertigoDataTier/pact/ddm_sm4_20260810/selected/r32_b4_uncentered/submission/archive.zip`, **182,364 B**, SHA-256 `6e6efb1154c763b49f713ce8fcaad4ee111425aaddd55b5566a4638b5429592e`.
- The independent staged receiver reconstructed state SHA-256 `5063e24b2c2374aff44db3f50eb8e000908034a54512dd4cbf3108d5aa59c01e`, exactly matching the prior SM3/CP2 r32 state.
- Reused full-RAW evidence: `/Volumes/VertigoDataTier/pact/ddm_cp2_20260810/pointwise_lowrank_r32__temporal_reversion/receiver_parseback/staging/0.raw`, **3,662,409,600 B**, SHA-256 `46ca24e7004c5a3ea42a118981a4fdf6a523e9d5b56cf6baff4444a062176f32`.
- Reused implementation provenance: `experiments/ddm_sm3_semantic_representation.py@d3650d6c68764385cad2d32faa394af7c87360c6` and `experiments/ddm_cp2_composition_receiver_and_harness.py@58d270898002cde052b4ad34506b14984db06d49`.

The new full inflate was intentionally not fired: the selected decoded state is byte-identical to a retained, full-inflated, exact-local-scored negative. Repeating that 1,086-second decode would only reproduce settled frames and would violate the operating contract's recall-before-rerun rule.

## Scorer disposition

**FOLDED-DEAD.** No scorer queue row was written. The arm owns no scorer slot, and its unique matched-budget optimum is the already-refuted r32 state rather than a new parity survivor.

## RECALL EVIDENCE

Stores consulted before execution:

- Governing state: `CLAUDE.md`, byte-identical `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the charter, and the common contract.
- Canonical experiment stores: `.omx/research/RESULTS_INDEX.md`, `.omx/research/CANONICAL_EXPERIMENT_DAG.md`, `.omx/research/CANONICAL_EQUATIONS.md`, `.omx/research/ddm_sm3_20260810/`, `.omx/research/ddm_main_paired_eval_20260810/`, and the bounded `.omx/research` / `.omx/state` corpus search.
- Real implementations and receivers: `experiments/ddm_sm3_semantic_representation.py`, `experiments/ddm_cp2_composition_receiver_and_harness.py`, `experiments/ddm_cp2_runtime/sm3r_receiver.py`, the retained CP2 submission receiver, and `upstream/evaluate.py` read-only.
- Canonical anti-pattern store: `src/tac/canonical_anti_patterns/`, especially `quantize_then_svd_corrupted_low_rank_v1`.

Representative recall commands:

```text
rg -n --glob '*.md' --glob '*.json' --glob '*.py' 'quantize_then_svd_corrupted_low_rank_v1|canonical_lowrank_factors|pointwise_lowrank_r32|factor quantization|row.center|centering' .omx/research .omx/state experiments src/tac
rg -n 'ddm_sm4|lowrank mechanism|r16-int8|r32-int4' .omx/research/RESULTS_INDEX.md .omx/research/CANONICAL_EXPERIMENT_DAG.md .omx/research/CANONICAL_EQUATIONS.md .omx/state/main_hot_state.md
shasum -a 256 /Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt /Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v2/retained/temporal_reversion/archive.zip
```

Recall changed the plan in three concrete ways. First, SM3 already does SVD on the real fp32 matrix before quantizing factors, so the quantize-then-SVD anti-pattern was ruled out rather than reimplemented. Second, SM3 had already shown that weight L2 is not a score selector, so weight error was used only as the charter's mechanism discriminator and the prior exact-state r32 result remained the adoption authority. Third, fresh hot state corrected the bankable token delta to **-2,416 B**; the non-additive `hp3` -8 B was not smuggled into this archive ledger.

No newer directive file was found in the bounded last-24-hour directive scan. No prior PR130 rank-by-precision-by-centering grid was found in the named canonical stores; this 410-cell retained grid is the new evidence for that scope.

## Commit custody

The canonical commit serializer was invoked with post-edit SHA-256 and `base-content-sha256=new` for all five files. It failed before staging because Git could not create an object-database temporary file: `error: unable to create temporary file: Operation not permitted`. The staged index remained empty. The implementation and receipts are therefore verified but uncommitted; the operator handoff must use the post-edit manifest in the final response.

## LIVE-HYPOTHESES

- A heterogeneous per-matrix rank/precision allocation may modestly beat global r32 at the same total bytes because the five measured r16/r32 degradation ratios differ from 1.327x to 1.435x. This is plausible as a constrained allocation improvement, but the 3.962x gap to direct q4 makes a large rescue unlikely.
- A low-rank core plus a separately coded structured tail could outperform pure truncation if the discarded singular tail has compressible spatial or block structure. The current grid measured only pure low-rank factors and row means; it did not test a retained residual grammar.

## DEAD-ENDS

- Uniform r16-int8 as the equal-law rescue is closed: it is worse in aggregate and on all 5/5 tensors, and its real archive is also larger.
- Stored fp16 row centering as the missing mechanism is closed at this budget: its fixed-rank gain is under 1%, and spending its bytes forces rank down to a worse matched-budget cell.
- Re-decoding or re-scoring the selected r32 state is closed: the staged state is byte-identical to the retained candidate already measured at odd-frame MAE 67.71 and local exact advisory S 7.4924.
