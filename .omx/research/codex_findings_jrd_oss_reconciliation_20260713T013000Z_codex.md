---
title: JRD and progressive-quantization OSS reconciliation
date_utc: 2026-07-13T01:30:00Z
lane_id: lane_449_master_oss_reconciliation_20260713
task_id: v9_jrd_coeff_prefix_probe_20260712
review_status: recovery-written-UNREVIEWED; task-hook-corrected; fresh-eyes-reviewed(0)
score_claim: false
pointer_moved: false
research_only: true
---

# JRD OSS reconciliation

## Outcome

**DERIVED · recovery-written-UNREVIEWED:** the landed JRD probe is an exact coefficient-prefix rate postprocessor, not an implementation of *The Last Byte*. No import-eligible JRD/Last-Byte, Li dead-zone, or PLONQ source was located. The one related official repository found, DeepHQ, assigns its relevant progressive-quantization code research/non-commercial patent terms. No source was copied, no provider was changed, and no enriched replay was admitted.

**DERIVED · recovery-written-UNREVIEWED:** JRD remains **NO-GO for #449 frozen-SegNet throughput**, with `verdict_scope=frozen-SegNet forward-plus-backward replacement`. The existing postprocessor removes no teacher forward or backward call. This does not negate its separate pair-0 rate-fixture result.

**MEASURED · fresh-eyes-reviewed(3):** the sealed local advisory fixture reduced the packet ZIP from 83,905 to 81,154 bytes while its exact pair-0 values changed from `d_seg=0.023157755533854168`, `d_pose=116.59830629690003` to `d_seg=0.0218505859375`, `d_pose=92.42743674059255`. Its fixture verdict remains GO. Its V9/v8 task verdict remains NEEDS-MORE because an eligible sealed non-live payload was missing or unresolved. `score_claim=false`; no contest axis was run.

## STORES CONSULTED

STORES CONSULTED: one `tools/corpus_query.py` query loaded research (5715), equations (622), memory (1893), DAG (505), council (277), tasks (96), and docs (92); the sealed JRD receipt and adversarial-review receipt; the current JRD implementation and harness; the AAAI article and PDF; IEEE DOI metadata; the PLONQ arXiv abstract and PDF; the DeepHQ repository and its actual LICENSE. Deliberately not loaded: the protected live V9 run, the active PR110 run contents, the full 5 GB GT cache, paid GPU surfaces, and provider actuation.

## Three-column reconciliation

| Surface | Ours, clean-room | Primary reference | What the conservative pass missed |
|---|---|---|---|
| JRD / last-safe search | Enumerates int8 coefficient prefixes, parses the receiver, evaluates exact through-R Seg/Pose, and admits a component-safe last-safe packet. | Xie et al. train MVR-Net to predict a frame encoding-QP map in one inference pass. No official repository or license was located. | The two methods are not algorithmic equivalents. MVR dataset annotation, QP-map supervision, and the learned dual-path predictor are absent. Those are a separate learned encoder-control project, not missing coefficient-prefix mechanics. |
| Dead-zone truncation | Uses uniform low-plane clearing plus a locally derived Laplace-motivated sign-magnitude dead zone, with a tested nested chain. | Li et al. describe learned progressive compression with dead-zone quantizers. No official repository or license was located. DeepHQ has related learned hierarchical quantization under restricted terms. | The exact Li et al. construction was not imported. The local `b*k*ln(2)` threshold must remain labeled DERIVED clean-room. |
| Progressive order / entropy | Sorts individual proposals by measured ZIP bytes saved, then exact-replays the combined packet under component-safe gates. It re-encodes the full int8 packet. | PLONQ uses nested quantization grids, conditional refinement coding, and rate-distortion latent ordering. DeepHQ uses learned hierarchical quantization and progressive coding. | There is no embedded refinement stream, conditional-probability telescoping, or delta-distortion/delta-rate latent order. These require a new archive grammar and are not drop-in coefficient mutations. |

## Source and license evidence

- Wuyuan Xie, Zhenming Li, Ye Liu, Jian Jin, Yun Song, and Miaohui Wang · 2026 · *The Last Byte: Learning Just Enough for Machine-Oriented Image Compression* · DOI `10.1609/aaai.v40i19.38635`. The [official AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/38635) resolves to that title and authors. The official page offers the paper and video/slides, not a code link. Exact-title, DOI, MVR-Net, author, and PDF searches did not locate an official source repository. This is `NOT_LOCATED`, not proof that no source exists.
- Shaohui Li, Han Li, Wenrui Dai, Chenglin Li, Junni Zou, and Hongkai Xiong · 2023 · *Learned Progressive Image Compression With Dead-Zone Quantizers* · DOI `10.1109/TCSVT.2022.3229701`. The [DOI](https://doi.org/10.1109/TCSVT.2022.3229701) resolves to the named IEEE paper. Exact-title, DOI, author, GitHub, and source-code searches did not locate an official repository. This is `NOT_LOCATED`, not proof of nonexistence.
- Yadong Lu, Yinhao Zhu, Yang Yang, Amir Said, and Taco S. Cohen · 2021 · *Progressive Neural Image Compression with Nested Quantization and Latent Ordering* · arXiv `2102.02913`; DOI `10.1109/ICIP42928.2021.9506026`. The [official arXiv page](https://arxiv.org/abs/2102.02913) resolves to the title and describes nested grids plus rate-distortion latent ordering. No official implementation was located.
- Jooyoung Lee, Se Yoon Jeong, and Munchurl Kim · 2025 · *DeepHQ: Learned Hierarchical Quantizer for Progressive Deep Image Coding* · arXiv `2408.12150`. The [official arXiv page](https://arxiv.org/abs/2408.12150) resolves to that title and author list. The [actual repository license](https://github.com/JooyoungLeeETRI/DeepHQ/blob/main/LICENSE) is dual: inherited TCM components are MIT, while hierarchical quantization, progressive coding, and learned quantizers are research/non-commercial and patent-protected. Those are the relevant components, so nothing was copied.

A single required shallow-clone attempt was made:

```text
git clone --depth 1 https://github.com/JooyoungLeeETRI/DeepHQ.git experiments/results/jrd_oss_reconciliation_20260713T011930Z/oss_sources/DeepHQ
rc=128
fatal: unable to access repository: Could not resolve host: github.com
```

## Re-measurement boundary

**UNKNOWN · recovery-written-UNREVIEWED:** there is no OSS-enriched exact delta. No implementation passed the source, license, and applicability gate, so rerunning the pair-0 evaluator would have reproduced the old program and falsely implied an enrichment comparison. The evidence-backed delta is therefore `UNKNOWN_NOT_MEASURED`, not zero.

The retained control law is an **event-conditioned tested predicate**: admit a nested candidate only when receiver parse-back succeeds, archive bytes strictly decrease, and both exact through-R component debts do not exceed their sealed baseline values. The named recess measurement is an eligible sealed non-live V9/v8 payload evaluated end to end; it remains unavailable in the sealed receipt.

## Triality and task boundaries

No lever, DSL term, or measured equation was added in this reconciliation. The existing JRD canonical equation
remains the measured-finding leg, and the DSL leg remains N/A-with-reason for an offline receiver/rate oracle.
The parent later applied an append-only task note and one master DAG FEED. The corrected task-hook and DAG
candidates remain beside the receipt at:

- `experiments/results/jrd_oss_reconciliation_20260713T011930Z/dag_feed_candidate.md`
- `experiments/results/jrd_oss_reconciliation_20260713T011930Z/task_hook_candidate.json`

The active PR110 run and its code were not touched.

## Verification and own round 1

- **MEASURED:** focused pytest passed 52 tests in 2.56 seconds, `rc=0`.
- **MEASURED:** Ruff returned “All checks passed!”, `rc=0`.
- **MEASURED:** `py_compile` returned no output, `rc=0`.
- **MEASURED:** both staged JSON files parsed with `jq -e`, `rc=0` before the review amendment.
- **REVIEW STATUS:** this reconciliation is `recovery-written-UNREVIEWED; task-hook-corrected;
  fresh-eyes-reviewed(0)`. The parent corrected the staged task operation after the subagent review, preserving
  blocked/green and `eligible_nonlive_v9_v8_payload_missing_or_unresolved`; that post-review change reset the
  counter. Final fresh-context provenance is carried only by the master verifier receipts.
