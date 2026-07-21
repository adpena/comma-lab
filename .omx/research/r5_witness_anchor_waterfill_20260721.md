# Task #578-R5 — exact ep725 witness reverse-waterfill

**Date:** 2026-07-21

**Lane:** `r5_witness_anchor_waterfill` · `research_only=true`

**Authority:** `[macOS-CPU advisory]`, full n600, CPU Torch hard oracle, batch 32, seed 1234

**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

**MAIN landing review:** **REQUIRED** before any merge, adoption, dispatch, or pointer action.

## Verdict

The exact shipped archive does **not** reproduce the checkpoint-side `d_seg=0.003457972208658854` premise. The hash-verified 83,838-byte archive measures `d_seg=0.0035127175506204367`, `d_pose=127.35957336425781`; the scope is this exact receiver output on macOS CPU, not a family negative and not a contest score claim.

Of the receiver-closed streams tested, only the bounded #336 `out_tex.weight` one-bit prefix mutation passes the measured waterfill. It yields composed v5 `(d_seg=0.003522830316796899, d_pose=127.03333282470703, bytes=83,827)`, advisory `S=36.04983575381022`, `delta S=-0.04473334744707813` versus the remeasured anchor. It does **not** justify paid exact evaluation: the non-comparable numeric gap to the contest-CPU pointer is `35.85875292961022`, with Pose accounting for `99.34696718054592%` of that gap.

Canonical composed receipt: `/Volumes/VertigoDataTier/pact/evidence/r5_waterfill_20260721/receipt_v3.json`, SHA-256 `0c471598bd1ad9204488d4ed69705900d986c97be76e805a166695e8fa69ee8a`.

## D1 — anchor custody and hard reproduction

The archive was found; no rebuild was needed.

| field | value | status |
|---|---:|---|
| archive | 83,838 B; SHA-256 `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3` | MEASURED bytes |
| decoded `0.raw` | 3,662,409,600 B; SHA-256 `8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae` | MEASURED bytes |
| checkpoint | ep725 EMA; SHA-256 `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef` | custody cross-reference only |
| `d_seg` | `0.0035127175506204367` official float32; `0.0035127173529730903` exact argmax rational | MEASURED through R |
| `d_pose` | `127.35957336425781` | MEASURED through R |
| advisory S | `36.0945691012573` | DERIVED from the three measured terms |
| checkpoint-to-shipped `delta d_seg` | `+0.000054745341961582455` | DERIVED; premise falsified |

D1 receipt SHA-256: `ca3a8ad38e3004219627318e2764fe59519aabb855e06d8f8944a1e779384480`. It preserves 19 fsync checkpoint rows, all scorer hashes, raw/source hashes, exact argv, and 414,377 write-once receiver-miss coordinates.

### Per-class Seg split

| class | mismatch pixels | contribution to all-pixel `d_seg` | conditional error |
|---|---:|---:|---:|
| Road | 138,540 | 0.0011744181315104167 | 0.005054905047582112 |
| Lane | 148,880 | 0.0012620713975694444 | 0.21556848078373797 |
| Undriv | 52,708 | 0.0004468112521701389 | 0.0009023290285247112 |
| Movable | 52,218 | 0.000442657470703125 | 0.0357577936418263 |
| MyCar | 22,031 | 0.00018675910101996528 | 0.0007345255445419511 |

## D2 — measured stream dispositions

Waterfill law: `lambda_star = 25 / 37,545,489 = 6.658589531221714e-7 S/B`, via `tac.canonical_equations.day_consolidation_laws_20260720.RATE_PRICE_S_PER_BYTE`. Admit only if measured `delta_nonrate_S + lambda_star * delta_bytes < 0`.

| stream | candidate triple `(d_seg, d_pose, B)` | `delta S` vs D1 | disposition / verdict scope |
|---|---|---:|---|
| #453 JRD transferred prefix | `(0.0053123729303479195, 128.35986328125, 79,008)` | `+0.316621139582665` | **REJECT**; the previous fixture does not transfer at zero distortion to this exact payload |
| #336 cheapest bounded requant | `(0.003522830316796899, 127.03333282470703, 83,827)` | `-0.04473334744707813` | **ADMIT**; exact singleton receiver-closed row, not an exhaustive bit curve |
| #140 low-rank dxi | no stream | `0` | **N/A** on this manifest: `pose_sidecar=false`; not a family negative |
| #400 pair-local polish | not actuated | — | **REFUSE**: `self_orient=true`; current Seg-only acceptance cannot protect joint Pose |
| R3 description donor | overlap `33,787 / 414,377 = 0.08153686136054848` | impossible best case `+0.09134352513598683` | overlap gate passes, but rate loses even if every overlap were fixed; RGB inverse-R binding also absent |

The #453 and #336 full-score receipt SHA-256 values are respectively `8a0f7e2c4e0150fc2fc73af84ea0e491f1aea86aef7fb4f9a2802f86942229d6` and `359283ef0e1f7c060e1573fa305e453b9bfefdcf0c20e963792550770d38fde6`.

### #557 coder race

The exact logical signed-int8 sections were measured with fully framed coders. Current joint LVLS1 inner streams are 61,598 B base + 21,010 B code. Isolated-section Brotli totals 82,842 B, IID arithmetic 101,949 B, and spatial-context arithmetic 130,330 B. Lossy block-FP weight bytes plus the unchanged code stream total 24,385 B, but there is no current LVLS1 block-FP parser/consumer and quantization distortion is unsettled.

Verdict: `BLOCKED_NO_LVLS1_CONTEXT_OR_BLOCK_FP_PARSER_CONSUMER`. No archive, full-decode, or rate-saving claim is made from unbound coder bytes. Receipt SHA-256: `616b217a12bb717ec6cb9544a7185bafc6c6d82518f2701b4679144f103a4727`.

## D3 — interaction closure and composed v5

Exactly one receiver-closed stream is admitted. Therefore the complete pairwise interaction matrix has zero pairs; union-once is the measured singleton itself and no commutator remains owed. This is not an additive approximation.

| v5 term | score units | share of advisory v5 |
|---|---:|---:|
| Seg | 0.3522830316796899 | 0.9772111975363327% |
| Pose | 35.64173576366716 | 98.86795603472359% |
| Rate | 0.055816958463372264 | 0.15483276774006605% |
| **total** | **36.04983575381022** | **100%** |

Composed archive: `/Volumes/VertigoDataTier/pact/evidence/r5_waterfill_20260721/d2_requant_cheapest/archive.zip`, 83,827 B, SHA-256 `d2ad27cf1c6a5e34482d04c0f08dfdb9d62c60ccfdeccf799381a5b4a5d8cbde`. It is a measured local candidate, not a submission archive claim.

## D4 — honest pointer gap

The pointer components are `d_seg=0.00055961`, `d_pose=0.00002942`, and 177,169 B from canonical equation `clickpolish_exact_gated_discrete_latent_ratchet_v1`; they recompute `S=0.19108282419209976`.

| component | v5 minus pointer score units | percentage of numeric gap |
|---|---:|---:|
| Seg | +0.2963220316796899 | +0.826358998767461% |
| Pose | +35.62458350434076 | +99.34696718054592% |
| Rate | -0.06215260640232972 | -0.17332617931337416% |
| **total** | **+35.85875292961812** | **100%** |

Axes are not promotion-comparable. The remaining binding axis is overwhelmingly Pose distortion, not bytes. **No paid row is justified** by v5; the pointer remains unchanged.

## Resumability, storage, and triality

- All three full n600 evaluations used preserved per-batch JSONL checkpoints and idempotent write-once mismatch chunks. A crash loses at most one batch.
- All 3.66 GB decoded raws and small receipts remain on `/Volumes/VertigoDataTier/pact/evidence/r5_waterfill_20260721/`; nothing was deleted or moved. The preflight storage plan passed. The receipts preserve original paths, byte counts, hashes, argv, scorer hashes, and false-authority flags.
- **DSL leg:** no trainer/runtime controller or invented flag was added; the landing is a research-only measurement adapter over the existing LVLS1, JRD, coder, and canonical-score interfaces.
- **DAG leg:** `.omx/research/r5_witness_anchor_waterfill_DAG_FEED_20260721.md` records the exact gates and next settling measurements.
- **Equation leg:** score uses `tac.contest_score.compute_contest_score`; rate price uses `RATE_PRICE_S_PER_BYTE`; pointer components use `clickpolish_exact_gated_discrete_latent_ratchet_v1`; checkpoint-side distortion is cited only as the falsified premise.
- **Pointer delta:** none.

## Tests and review

`11 passed`; Ruff clean; `py_compile` clean; `git diff --check` clean. The two Python surfaces and both focused test files received explicit `review_tracker` passes `r5_pass1` and `r5_pass2`.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 specs; `reports/latest.md`; lane/task/subagent/inbox canonical state; ep725 duty ticket and yhat R-D ladder receipts; #400/#453/#401/#402/#140/#336/#311/#557/#553 code and memos; exact anchor archive/checkpoint; R3 donor packet; SSD receipts listed above; reuse manifest beside this memo.
