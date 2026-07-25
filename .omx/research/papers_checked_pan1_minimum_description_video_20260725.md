# Papers checked — PAN1 minimum-description frozen-scorer video sweep

Date: 2026-07-25. Scope: bounded 2025–2026 primary-source sweep after
deduplicating the existing `.omx/research/papers_checked_*` ledger. No launch,
checkout, dependency install, or source adoption occurred.

## New, directly relevant rows

| Source | Primary claim consumed | Honest transfer to this repository | Disposition |
|---|---|---|---|
| [RL-RC-DoT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html) | Task-aware macroblock QP control improves downstream-task rate allocation without requiring the task network at inference. | Independent support for evaluator-cell-aware allocation. It does **not** replace exact frozen-scorer admission and is weaker than the existing per-cell/per-stream split because its vehicle is a standard codec. | `CORROBORATES_ALLOCATION_ONLY`; no new build. |
| [Neural Video Compression with Context Modulation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.html) and [official code](https://github.com/Austin4USTC/DCMVC) | Flow orientation plus reference-conditioned context compensation removes irrelevant propagated context; the paper reports a 10.1% rate saving over DCVC-FM. | Directly motivates racing the 100,099-byte stream against xi/reference-conditioned regeneration rather than another outer coder. The published percentages are `MEASURED-ELSEWHERE`, never our forecast. | `ADOPT_MEASUREMENT_SHAPE` for PAN1-B04/la1 follow-on. |
| [Ultra-lightweight Neural Video Representation Compression, arXiv 2512.04019](https://arxiv.org/abs/2512.04019) | Multi-scale grids plus an octree context model replace slow autoregressive entropy coding; reported decode acceleration and BD-rate gains are external. | The octree/context factorization is a concrete alternative representation for the dominant v15 stream. It must be retargeted to the exact same semantic object and exact member bytes; no PSNR result transfers. | `ADOPT_BYTE_ONLY_PROTOTYPE_SHAPE`. |
| [InnVC, arXiv 2606.13957](https://arxiv.org/abs/2606.13957) | An invertible main path plus compact implicit conditioning and scheduled channel masking separates correlated content from fine details. | Strongest new structural seed: make the rule-118 generic path invertible/video-independent, count only compact content conditioning, and test progressive masks by exact scorer survival. The current paper has no frozen-scorer or contest-rule authority. | `ADOPT_HONEST_FORK_PROBE`, not a vehicle decision. |

## Negative and dedup findings

- Task-aware compression, codec-aware VO, knowledge distillation, Cells2Pixels,
  and general INR families already have papers-checked entries; they were not
  re-opened.
- None of the new sources studies this contest's exact objective, frozen
  SegNet argmax cells, PoseNet YUV6 MSE, rule-118 charge boundary, or exact
  archive bytes. Their numerical results are therefore literature evidence,
  not predicted score movement.
- No newly found source resolves PAN1-B01/B02: that issue is internal evaluator
  arithmetic, already settled by `upstream/evaluate.py` and the campaign
  launcher.

Pointer `0.19108282419209976 [contest-CPU]` UNMOVED. `research_only=true`;
`score_claim=false`; `main_review_required=true`.
