# ddm_mz2 — frozen semantic-section representation attack

**Date:** 2026-08-15  
**Object:** exact e480b RX1M semantic section, archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`  
**Measurement axis:** `[macOS-CPU advisory; scorer-free current-e480b section representation]`  
**Result receipt:** `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/FINAL_RESULT.json`, SHA-256 `e1b47a550c700e0c0544ec479d70de4e68878dce7c31aee3aafd11cb32762b58`

## Result first

The current F12 semantic representation remains the exact-state winner. The frozen receiver consumes all **38/38** semantic tensors. None is a structural zero/one that can be derived for free; all **16/16** quantized matrices are numerically full-rank at their matrix-shape bound; none has a duplicate or zero row. Four strict fixed-schema exact-state forms—dense, zero-sparse, row-dictionary, and their hybrid—each rebuilt to **183,842 B**, or **+340 B** versus the 183,502 B control. This closes those exact-state forms on this instance; it does not kill all possible exact parametrizations.

The current-state sensitivity allocation from SD1 does produce a byte-distinct archive: **182,679 B**, **−823 B**, SHA-256 `b3b38b0672036814ce1804faaee0c3eeffaf61caf94edcc39ea7161044f04430`, with a rate-only projection of `−0.0005480019 S`. Its strict measurement decoder reconstructs exactly the intended mixed-bit state. It changes 7,991/66,339 decoded parameters across exactly `frame_embed.weight` and the three selected FiLM matrices. It has **no current-vehicle Seg/Pose measurement**, no shipping SD1M receiver, and therefore is **NOT_ADMITTED**. The prior PR130 SD1 n600 result is ancestor evidence only and is not transferred.

Six current-state structured Film-row sparsity cells were also byte-closed and retained. They save **130–2,051 B**, but no current score exists:

| Candidate | Archive bytes | Delta vs e480b | Changed decoded parameters | Disposition |
|---|---:|---:|---:|---|
| mixed q3/q4 | 182,679 | −823 | 7,991 | QUEUED-WITH-A-FIRE-ORDER |
| Film keep 87% | 183,372 | −130 | 548 | QUEUED after mixed-bit verdict |
| Film keep 75% | 183,031 | −471 | 1,051 | QUEUED after mixed-bit verdict |
| Film keep 62% | 182,754 | −748 | 1,598 | QUEUED after mixed-bit verdict |
| Film keep 50% | 182,437 | −1,065 | 2,093 | QUEUED after mixed-bit verdict |
| Film keep 37% | 181,978 | −1,524 | 2,646 | QUEUED after mixed-bit verdict |
| Film keep 25% | 181,451 | −2,051 | 3,143 | QUEUED after mixed-bit verdict |

Even the largest scorer-unvalidated reduction is only 13.5% of the **15,153 B** fixed-distortion saving required for sub-0.15. No exact or contest score ran. The pointer did not move.

## Measured boundaries

- **Receiver-consumption bijection — measured, 38/38.** F12 restores the canonical WANS records, all names and shapes match `SemanticTokenRenderer(96).state_dict()`, the real `strict=True` load succeeds, and deleting each key independently makes strict load refuse. The mapped embeddings, coordinate projection, all four full blocks, and head are traversed by the renderer topology. No unread semantic tensor was found in this exact receiver.
- **Derive at decode — instance closed for structural constants, 0/38.** No consumed tensor is identically zero or one. Video-derived constants were not relabeled as free code. Seed/config reconstruction beyond those generic constants was not asserted.
- **Exact low-rank/factor signal — instance negative.** All 16 int4 code matrices are numerically full-rank at their shape bound; all have zero duplicate rows and zero zero-rows. This supports the prior SM3/SM4 result and supplies no exact factorization. Numerical SVD is diagnostic, not a proof against every algebraic representation.
- **Exact zero-sparse/row-dictionary forms — instance closed.** Across 4 candidates × 16 quantized tensors, 0/64 tensor selectors chose sparse or row-dictionary storage over dense. Outer Brotli q11 still left every complete archive at 183,842 B.
- **Mixed precision — byte gate passed, score gate open.** The exact retained archive saves 823 B. It changes only the four intended q3 tensors; all q4 and fp16 tensors reconstruct identically. Admission still requires current receiver-closed n600 with net `ΔS < −3.5e-6`.
- **Structured sparsity — byte gate passed, score gate open.** The six-cell per-lever keep curve is retained and parse-backed. Only `blocks.{1,2,3}.film.weight` changes. No score conclusion is drawn from changed-weight counts.
- **Distilled smaller renderer — QUEUED-WITH-A-FIRE-ORDER.** An untrained width truncation would be fake. The real deterministic, resumable distillation sweep must wait until the untouchable e960 burn and governed scorer/trainer slot permit it.
- **Carrier second scope — QUEUED-WITH-A-FIRE-ORDER.** This unit exercised the charter's legal semantic-first reduction. Carrier work may fire only after semantic mechanisms terminate and must preserve the same decoded values while avoiding PK2/PK4/PS135B/MZ1 repeats.

## What was not measured

- No current-e480b SegNet or PoseNet result for any byte-distinct semantic candidate.
- No shipping-receiver decode of MZ2E, SD1M, or SM3R archive bytes.
- No contest-CPU or contest-CUDA evaluation and no paid dispatch.
- No distillation training and no new carrier representation.
- No claim that numerical full rank rules out all nonlinear or programmatic exact representations.

The charter did not claim the single full-n600 scorer slot, so the common contract requires scorer-free work plus a queue. `SCORER_QUEUE.json`, `STRUCTURED_SPARSITY_QUEUE.json`, `DISTILL_QUEUE.json`, and `CARRIER_QUEUE.json` each carry an owner, consumer store, and fire trigger. `T4_FIRE_ORDER.json` is **FOLDED** because no byte-distinct candidate is both shipping-receiver-closed and current-vehicle score-admitted. MAIN remains the only T4 owner; Modal was not used.

## RECALL EVIDENCE

The search covered `.omx/research/` memos and receipts, arm final messages, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, canonical equations from `tools/list_canonical_equations.py --json`, `.omx/state/main_hot_state.md`, and task/charter surfaces. Content queries included `semantic renderer`, `semantic weight`, `WANS`, `F12`, `lowrank`, `mixed q3`, `representation attack`, `frozen section`, `receiver consumption`, `SM3`, `SM4`, `SV3`, and `SD1`.

Findings beyond the charter seeds changed the plan:

- `ddm_sm3_20260810/SM3_FINDINGS.md` and its retained result already materialized eight PR130 semantic representations. This prevented rebuilding VQ, low-rank, and prune format machinery from scratch.
- `ddm_cp2_20260810/CP2_FINDINGS.md` had already proved real receiver closure for PR130 low-rank/VQ/mixed composition; that made receiver format reuse legitimate but did not transfer scores to current e480b.
- `ddm_sm4_20260810/SM4_FINDINGS.md` covered a 410-cell low-rank rank/bit/centering grid; the selected r32-int4 state was later refuted. That closed another low-rank sweep here.
- `ddm_sv3_20260810/SV3_FINDINGS.md` found low-rank and joint-VQ raw outputs catastrophically different, while mixed q3/q4 survived the cheap screen. This promoted only mixed precision into the first current-state score queue.
- `ddm_sd1_semantic_20260809/SD1_FINDINGS.md` supplied the per-tensor selected q3 allocation and ancestor n600 evidence. The current two-codeword FiLM difference means its score does not transfer; this unit rebuilt the allocation on current decoded bytes.
- `ddm_fd135_fractal_decomposition_20260810.md` proved the current 36,040 B semantic body is the F12 WANS state, with only two current semantic-code changes from PR130, and supplied the strict 38-tensor schema. It replaced checkpoint inference with direct current-byte decode.
- The receiver-consumption DAG surfaces identify #417 as the correct bijection gate. The canonical-equations search found general quantization/representation priors but no equation that overrides the direct current-byte identity and archive measurements.

## Custody and verification

All materialized bytes are under `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/`. The retention inventory covers **167 files / 8,183,925 B**, SHA-256 `156112d0a0b8caeec0f0a6eaedd3bc1d24e2d389b199dad2495324ebd6c2dbcc`; it excludes itself and `FINAL_RESULT.json` to avoid recursive hashes. It includes per-tensor decoded arrays, exact stored records and codes, every candidate semantic section, its Brotli stream, RX1M model, member `p`, archive, repeat archive, and all nonrecursive stage/queue receipts. Every repeat archive is byte-identical. The live e960 directory and checkpoints were not read or modified.

Verification:

- `4 passed` — `experiments/tests/test_ddm_mz2_frozen_section_representation_attack.py`
- payload-retention gate: `0` findings across the runner and test
- staged resume receipts: preflight, autopsy, exact race, score gate, finalize
- source base: `d28520ad26528be68db4e718789c74ede62fbf66`

## LIVE-HYPOTHESES

- **Current-state mixed q3/q4 may be a small real rate win.** It is plausible because the same selected allocation survived the prior current-lineage cheap screen and won an ancestor n600 semantic-leg trade, while the rebuilt e480b archive saves 823 real bytes. Only current receiver-closed Seg/Pose scoring can decide it.
- **Very light Film-row sparsity may offer a second small Pareto point.** Keep-87 changes only 548 parameters for 130 B; the six retained cells expose a monotone rate curve. It remains plausible but weaker than mixed precision because no prior n600 survivor exists.
- **A genuinely trained smaller renderer remains the only semantic mechanism here with potential for a multi-kilobyte cut.** Width reduction changes the architecture rather than recoding full-rank matrices, and could exceed the 2,051 B structured-sparsity ceiling. It is plausible only with real distillation, retained learned payloads, and receiver-closed scoring.
- **Carrier representation remains second-scope headroom.** CAP1 still explicitly represents 27,648 basis symbols and 7,200 coefficient values, so a different same-decode parametrization is conceivable; prior coding and pose-solve closures make only a non-repeated structural representation admissible.

## DEAD-ENDS

- **Unread-tensor deletion:** closed on this e480b receiver because all 38/38 fixed-schema keys are required by strict load and mapped into the traversed renderer.
- **Structural zero/one derivation:** closed on this e480b state because 0/38 tensors match those free generic constants.
- **Exact row-dictionary or zero-sparse semantic storage:** closed in the tested fixed-schema scope because 0/64 selectors beat dense and every full archive was +340 B.
- **Another pointwise low-rank/VQ sweep:** not retried; SM3, SM4, CP2, and SV3 already close the tested mechanisms, and the current matrices add a 16/16 full-rank, no-duplicate-row diagnostic.
- **Lossless recoding of F12:** not retried; MZ1 already closed the eight-way real-coder race.
- **Immediate T4 dispatch:** folded because no new archive has both a shipping receiver and a current-vehicle score.

Own-vehicle frontier remains **S = 0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]`**, archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`; pointer unmoved.
